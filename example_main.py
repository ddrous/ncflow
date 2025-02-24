### THIS IS THE MAIN SCRIPT TO TRAIN EITHER A NCF-T1 or -T2 ###

## Import all the necessary libraries
from ncf import *

## Set the seed for reproducibility
seed = 2026

## NCF main hyperparameters ##
context_pool_size = 6               ## Number of neighboring contexts j to use for a flow in env e
context_size = 1024                 ## Size of the context vector

nb_outer_steps_max = 2000           ## maximum number of outer steps when using NCF-T2
nb_inner_steps_max = 25             ## Maximum number of inner steps when using NCF-T2 (for both weights and contexts)
proximal_beta = 1e2                 ## Proximal coefficient, see beta in https://proceedings.mlr.press/v97/li19n.html
inner_tol_node = 1e-9               ## Tolerance for the inner optimisation on the weights
inner_tol_ctx = 1e-8                ## Tolerance for the inner optimisation on the contexts
early_stopping_patience = nb_outer_steps_max//1       ## Number of outer steps to wait before stopping early

## General training hyperparameters ##
print_error_every = 10              ## Print the error every n epochs
integrator = diffrax.Dopri5         ## Integrator to use for the learner
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":40000, "subdivisions":5}
init_lr = 1e-4                      ## Initial learning rate
sched_factor = 0.1                  ## Factor to multiply the learning rate by at after 1/3 and 2/3 of the total gradient steps

run_folder = None                   ## Folder to save the results of the run
train = False                       ## Train the model, or load a pre-trained model
save_trainer = True                 ## Save the trainer object after training
finetune = False                    ## Finetune a trained model

## Adaptation hyperparameters ##
adapt_test = True                   ## Test the model on an adaptation dataset
adapt_restore = False               ## Restore a trained adaptation model

init_lr_adapt = 5e-3                ## Initial learning rate for adaptation
sched_factor_adapt = 0.5            ## Factor to multiply the learning rate by at after 1/3 and 2/3 of the total gradient steps
nb_epochs_adapt = 1500              ## Number of epochs to adapt



#%%


if train == True:

    ## Create a folder for the runs
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside for the current run
    run_folder = './runs/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    os.mkdir(run_folder)
    print("Run folder created successfuly:", run_folder)

    # Save this main and the dataset gneration scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp dataset.py {run_folder}")

    # Save the nodax module files as well
    os.system(f"cp -r ../../nodax {run_folder}")
    print("Completed copied scripts ")

else:
    print("No training. Loading data and results from:", run_folder)

## Create a folder for the adaptation results
adapt_folder = run_folder+"adapt/"
if not os.path.exists(adapt_folder):
    os.mkdir(adapt_folder)

#%%

if train == True:
    # Run the dataset script to generate the data: meta-training and the corresponding evaluation (or download it from Gen-Dynamics)
    os.system(f'python dataset.py --split=train --savepath="{run_folder}" --seed="{seed}"')
    os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}"')

## Define dataloader for training
train_dataloader = DataLoader(run_folder+"train_data.npz", batch_size=-1, shuffle=True, key=seed)

## Useful information about the data - As per the Gen-Dynamics interface
nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

## Define dataloader for validation (for selecting the best model during training)
val_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)

#%%

## Define model and loss function for the learner
class Swish(eqx.Module):
    """Swish activation function"""
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)

class Augmentation(eqx.Module):
    """ Nueral Network for the neural ODE's vector field """
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


### Define the Taylor expantion about the context vector ###
class ContextFlowVectorField(eqx.Module):
    physics: eqx.Module
    augmentation: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics

    def __call__(self, t, x, ctxs):
        ctx, ctx_ = ctxs

        if self.physics is None:
            vf = lambda xi: self.augmentation(t, x, xi)
        else:
            vf = lambda xi: self.physics(t, x, xi) + self.augmentation(t, x, xi)

        gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
        scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

        # return vf(ctx_) + 1.0*gradvf(ctx_)                          ## If NCF-T1
        return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term     ## If NCF-T2

## Create the neural network (accounting for the unknown in the system), and use physics is problem is known
augmentation = Augmentation(data_size=2, int_size=64, context_size=context_size, key=seed)
vectorfield = ContextFlowVectorField(augmentation, physics=None)

print("\n\nTotal number of parameters in the model:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(vectorfield,eqx.is_array)) if x is not None), "\n\n")

## Define the context parameters for all environwemts in a single module
contexts = ContextParams(nb_envs, context_size, key=None)

## Define a custom loss function
def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s, key):

    ## Define the context pool using the Random-All strategy
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction
    term2 = jnp.mean(jnp.abs(ctx))              ## context regularisation
    term3 = params_norm_squared(model)          ## weight regularisation

    loss_val = term1 + 1e-3*term2 + 1e-3*term3

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)   ## The neural ODE integrator returns the number of steps taken, which are handy for analysis

## Finnaly, create the learner
learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=seed)


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
    ## Ordinary alternating minimsation to train the NCF-t1 model
    # trainer.train(nb_epochs=nb_total_epochs*(2**0), 
    #             print_error_every=print_error_every*(2**0), 
    #             update_context_every=1, 
    #             save_path=trainer_save_path, 
    #             key=seed, 
    #             val_dataloader=val_dataloader, 
    #             int_prop=1.0)

    ## Proximal alternating minimisation to train the NCF-t2 model
    trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max, 
                            nb_inner_steps_max=nb_inner_steps_max, 
                            proximal_reg=proximal_beta, 
                            inner_tol_node=inner_tol_node, 
                            inner_tol_ctx=inner_tol_ctx,
                            print_error_every=print_error_every*(2**0), 
                            save_path=trainer_save_path, 
                            val_dataloader=val_dataloader, 
                            patience=early_stopping_patience,
                            int_prop=1.0,
                            key=seed)
else:
    restore_folder = run_folder
    trainer.restore_trainer(path=restore_folder)



#%%

## Finetune the neural network weights of a trained model
if finetune:
    finetunedir = run_folder+"finetune_"+trainer.dataloader.data_id+"/"
    if not os.path.exists(finetunedir):
        os.mkdir(finetunedir)
    print("No training. Loading anctx_sd finetuning into:", finetunedir)

    trainer.dataloader.int_cutoff = nb_steps_per_traj

    opt_node = optax.adabelief(1e-7)
    opt_ctx = optax.adabelief(1e-7)
    trainer.opt_node, trainer.opt_ctx = opt_node, opt_ctx

    trainer.train(nb_epochs=24000, print_error_every=1000, update_context_every=1, save_path=finetunedir, key=seed)


#%%

## Test and visualise the results on a test dataloader (same as the validation dataset)
test_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)
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
    os.system(f'python dataset.py --split=adapt --savepath="{adapt_folder}" --seed="{seed*3}"');
    os.system(f'python dataset.py --split=adapt_test --savepath="{adapt_folder}" --seed="{seed*3}"');

## Adaptation of the model to a new dataset
if adapt_test:
    adapt_dataloader = DataLoader(adapt_folder+"adapt_data.npz", adaptation=True, data_id="170846", key=seed)           ## TRAIN
    adapt_dataloader_test = DataLoader(adapt_folder+"adapt_test_data.npz", adaptation=True, data_id="170846", key=seed) ## TEST

    ## Define the optimiser for the adaptation (optional)
    sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr_adapt,
                            boundaries_and_scales={nb_total_epochs//3:sched_factor_adapt, 2*nb_total_epochs//3:sched_factor_adapt})
    opt_adapt = optax.adabelief(sched_ctx_new)

    if adapt_restore == False:
        trainer.adapt_sequential(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every, save_path=adapt_folder)
    else:
        print("Save_id for restoring trained adapation model:", adapt_dataloader.data_id)
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)

    ## Evaluate the model on the adaptation test dataset
    ood_crit, _ = visualtester.test(adapt_dataloader_test, int_cutoff=1.0)

    visualtester.visualize(adapt_dataloader_test, int_cutoff=1.0, save_path=adapt_folder+"results_ood.png");