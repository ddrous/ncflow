#%%
### MAIN SCRIPT TO TRAIN A NEURAL CONTEXT FLOW ###

# %load_ext autoreload
# %autoreload 2

## Import all the necessary modules
from ncf import *

## Seed for reproducibility in JAX
seed = 2026

## NCF main hyperparameters ##
context_pool_size = 2               ## Number of neighboring contexts j to use for a flow in env e
context_size = 256                 ## Size of the context vector

nb_outer_steps_max = 1500           ## maximum number of outer steps when using NCF-T2
nb_inner_steps_max = 10             ## Maximum number of inner steps when using NCF-T2 (for both weights and contexts)
proximal_beta = 1e2                 ## Proximal coefficient, see beta in https://proceedings.mlr.press/v97/li19n.html
inner_tol_node = 1e-9               ## Tolerance for the inner optimisation on the weights
inner_tol_ctx = 1e-8                ## Tolerance for the inner optimisation on the contexts
early_stopping_patience = nb_outer_steps_max       ## Number of outer steps to wait before stopping early

## General training hyperparameters ##
print_error_every = 10              ## Print the error every n epochs
integrator = diffrax.Dopri5         ## Integrator to use for the learner
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":40000, "subdivisions":2}
init_lr = 1e-4                      ## Initial learning rate
sched_factor = 1.0                  ## Factor to multiply the learning rate by at after 1/3 and 2/3 of the total gradient steps
ncf_variant = 2                     ## 1 for NCF-T1, 2 for NCF-T2
taylor_order = ncf_variant          ## Taylor order for the neural ODE's vector field
print(f"NCF variant: NCF-t{ncf_variant}")

train = True                            ## Train the model, or load a pre-trained model
run_folder = None if train else "./"    ## Folder to save the results of the run
save_trainer = True                     ## Save the trainer object after training
finetune = False                        ## Finetune a trained model
data_folder = "./data/" if train else "../../data/"  ## Where to load the data from

## Adaptation hyperparameters ##
adapt_test = True                   ## Test the model on an adaptation dataset
adapt_restore = False               ## Restore a trained adaptation model
sequential_adapt = True

init_lr_adapt = 1e-4                ## Initial learning rate for adaptation
sched_factor_adapt = 1.0            ## Factor to multiply the learning rate by at after 1/3 and 2/3 of the total gradient steps
nb_epochs_adapt = 1500              ## Number of epochs to adapt


#%%

if run_folder==None:
    run_folder = make_run_folder('./runs/')
else:
    print("Using existing run folder:", run_folder)

adapt_folder = setup_run_folder(folder_path=run_folder, script_name=os.path.basename(__file__))


#%%

if train == True:
    # If no data is available in Gen-Dynamics: https://github.com/ddrous/gen-dynamics, generate it as below
    if not os.path.exists(data_folder+"train.npz") or not os.path.exists(data_folder+"test.npz"):
        os.system(f'python dataset.py --split=train --savepath="{data_folder}"')
        os.system(f'python dataset.py --split=test --savepath="{data_folder}"')

## Define dataloader for training
train_dataloader = DataLoader(data_folder+"train.npz", shuffle=True, key=seed)

## Useful information about the data - As per the Gen-Dynamics interface
nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

print("== Properties of the data ==")
print("Number of environments:", nb_envs)
print("Number of trajectories per environment:", nb_trajs_per_env)
print("Number of steps per trajectory:", nb_steps_per_traj)
print("Data size:", data_size)

## Define dataloader for validation (for selecting the best model during training)
val_dataloader = DataLoader(data_folder+"test.npz", shuffle=False)




#%%

## Define model and loss function for the learner

class NeuralNet(eqx.Module):
    layers_data: list
    layers_context: list
    layers_shared: list
    activations: list

    def __init__(self, data_size, int_size, context_size, key=None):
        keys = generate_new_keys(key, num=12)
        self.activations = [Swish(key=key_i) for key_i in keys[:7]]

        self.layers_context = [eqx.nn.Linear(context_size, context_size//4, key=keys[0]), self.activations[0],
                               eqx.nn.Linear(context_size//4, int_size, key=keys[1]), self.activations[1], eqx.nn.Linear(int_size, int_size, key=keys[2])]

        self.layers_data = [eqx.nn.Linear(data_size, int_size, key=keys[3]), self.activations[2], 
                            eqx.nn.Linear(int_size, int_size, key=keys[4]), self.activations[3], 
                            eqx.nn.Linear(int_size, int_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(2*int_size, int_size, key=keys[6]), self.activations[4], 
                              eqx.nn.Linear(int_size, int_size, key=keys[7]), self.activations[5], 
                              eqx.nn.Linear(int_size, int_size, key=keys[8]), self.activations[6], 
                              eqx.nn.Linear(int_size, data_size, key=keys[9])]

    def __call__(self, t, y, ctx):

        for layer in self.layers_context:
            ctx = layer(ctx)

        for layer in self.layers_data:
            y = layer(y)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)

        return y

## Define a loss function for one environment (one context)
def loss_fn_env(model, trajs, t_eval, ctx, all_ctx_s, key):

    ## Define the context pool using the Random-All strategy
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction loss
    term2 = jnp.mean(jnp.abs(ctx))              ## context regularisation
    # term3 = params_norm_squared(model)          ## weight regularisation

    # loss_val = term1 + 1e-3*term2 + 1e-3*term3
    loss_val = term1 + 1e-3*term2

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)


## Create the neural network (accounting for the unknown in the system), and use physics is problem is known
neuralnet = NeuralNet(data_size=2, int_size=64, context_size=context_size, key=seed)
vectorfield = SelfModulatedVectorField(physics=None, augmentation=neuralnet, taylor_order=taylor_order)
## Define the context parameters for all environwemts in a single module
contexts = ContextParams(nb_envs, context_size, key=None)

print("\n\nTotal number of parameters in the neural ode:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(vectorfield, eqx.is_array)) if x is not None))
print("Total number of parameters in the contexts:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(contexts, eqx.is_array)) if x is not None), "\n\n")

## Finnaly, create the learner
learner = Learner(vectorfield, contexts, loss_fn_env, integrator, ivp_args, key=seed)








#%%

## Define optimiser and train the model
nb_total_epochs = nb_outer_steps_max * nb_inner_steps_max
sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})
sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})

opt_node = optax.adam(sched_node)
opt_ctx = optax.adam(sched_ctx)

## Create the trainer
trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=seed)

#%%

trainer_save_path = run_folder if save_trainer == True else False

if train == True:
    if ncf_variant == 1:
        ## Ordinary alternating minimsation to train the NCF-t1 model
        trainer.train_ordinary(nb_epochs=nb_total_epochs, 
                                print_error_every=print_error_every, 
                                update_context_every=1, 
                                save_path=trainer_save_path, 
                                key=seed, 
                                val_dataloader=val_dataloader, 
                                int_prop=1.0)
    elif ncf_variant == 2:
        ## Proximal alternating minimisation to train the NCF-t2 model
        trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max, 
                                nb_inner_steps_max=nb_inner_steps_max, 
                                proximal_reg=proximal_beta, 
                                inner_tol_node=inner_tol_node, 
                                inner_tol_ctx=inner_tol_ctx,
                                print_error_every=print_error_every, 
                                save_path=trainer_save_path, 
                                val_dataloader=val_dataloader, 
                                patience=early_stopping_patience,
                                int_prop=1.0,
                                key=seed)
    else:
        raise ValueError("NCF variant must be 1 or 2")

else:
    restore_folder = run_folder
    trainer.restore_trainer(path=restore_folder)


#%%

## Finetune the neural weights of a trained model
if finetune:
    finetunedir = run_folder+"finetune/"
    if not os.path.exists(finetunedir):
        os.mkdir(finetunedir)
    print("No training. Loading and finetuning into:", finetunedir)

    trainer.dataloader.int_cutoff = nb_steps_per_traj

    opt_node = optax.adabelief(1e-7)
    opt_ctx = optax.adabelief(1e-7)
    trainer.opt_node, trainer.opt_ctx = opt_node, opt_ctx

    trainer.train(nb_epochs=2400, print_error_every=1000, update_context_every=1, save_path=finetunedir, key=seed)


#%%

## Test and visualise the results on a test dataloader (same as the validation dataset)
test_dataloader = DataLoader(data_folder+"test.npz", shuffle=False)
visualtester = VisualTester(trainer)

ind_crit = visualtester.test(test_dataloader, int_cutoff=1.0)

if finetune:
    savefigdir = finetunedir+"results_in_domain.png"
else:
    savefigdir = run_folder+"results_in_domain.png"
visualtester.visualize(test_dataloader, int_cutoff=1.0, save_path=savefigdir);







#%%

## Create a new dataset for adaptation and another one to test the model on it (or download it from Gen-Dynamics)
if adapt_test and not adapt_restore:
    if not os.path.exists(data_folder+"ood_train.npz") or not os.path.exists(data_folder+"ood_test.npz"):
        os.system(f'python dataset.py --split=adapt --savepath="{data_folder}"');
        os.system(f'python dataset.py --split=adapt_test --savepath="{data_folder}"');

## Adaptation of the model to a new dataset
if adapt_test:
    adapt_dataloader = DataLoader(data_folder+"ood_train.npz", adaptation=True, key=seed)               ## TRAIN
    adapt_dataloader_test = DataLoader(data_folder+"ood_test.npz", adaptation=True, key=seed)           ## TEST

    ## Define the optimiser for the adaptation (optional)
    sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr_adapt,
                            boundaries_and_scales={nb_total_epochs//3:sched_factor_adapt, 2*nb_total_epochs//3:sched_factor_adapt})
    opt_adapt = optax.adabelief(sched_ctx_new)

    if adapt_restore == False:
        if sequential_adapt:
            trainer.adapt_sequential(adapt_dataloader, 
                                    nb_epochs=nb_epochs_adapt, 
                                    optimizer=opt_adapt, 
                                    print_error_every=print_error_every, 
                                    save_path=adapt_folder,
                                    key=seed)
        else:
            trainer.adapt_bulk(adapt_dataloader, 
                                nb_epochs=nb_epochs_adapt, 
                                optimizer=opt_adapt, 
                                print_error_every=print_error_every, 
                                save_path=adapt_folder,
                                key=seed)
    else:
        print("Restoring trained adapation model")
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)

    ## Evaluate the model on the adaptation test dataset
    ood_crit, _ = visualtester.test(adapt_dataloader_test, int_cutoff=1.0)

    visualtester.visualize(adapt_dataloader_test, int_cutoff=1.0, save_path=adapt_folder+"results_ood.png");




#%%
## After training, copy nohup.log to the runfolder
try:
    __IPYTHON__         ## in a jupyter notebook
except NameError:
    os.system(f"cp nohup.log {run_folder}")
