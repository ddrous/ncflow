#%%
# import os
# %load_ext autoreload
# %autoreload 2

## Do not preallocate GPU memory
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"platform\"'

from ncf import *
# jax.config.update("jax_debug_nans", True)

## Execute jax on CPU
# jax.config.update("jax_platform_name", "cpu")





#%%


seed = 2026
# seed = int(np.random.randint(0, 10000))

context_pool_size = 1               ## Number of neighboring contexts j to use for a flow in env e
context_size = 256//1
nb_epochs = 1200*1
init_lr = 3e-4
lr_factor = 0.1

print_error_every = 400

train = True
# run_folder = "./runs/03082024-024220-Toy/"      ## Run folder to use when not training
reuse_run_folder = False

save_trainer = True
finetune = False

adapt = True
adapt_huge = False

integrator = diffrax.Dopri5
# integrator = RK4
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":4000, "subdivisions":10}


#%%


if train == True:

    # check that 'tmp' folder exists. If not, create it
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside 'tmp' whose name is the current time
    run_folder = './runs/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    # run_folder = "./runs/21112024-123322-OPE-SampleEfficiency/"
    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
    print("Run folder created successfuly:", run_folder)

    # Save the run and dataset scripts in that folder
    script_name = os.path.basename(__file__)
    os.system(f"cp {script_name} {run_folder}")
    os.system(f"cp dataset.py {run_folder}")

    # Save the nodax module files as well
    os.system(f"cp -r ../../nodax {run_folder}")
    print("Completed copied scripts ")

else:
    # run_folder = "./runs/22022024-112457/"  ## Needed for loading the model and finetuning TODO: opti
    print("No training. Loading data and results from:", run_folder)

## Create a folder for the adaptation results
adapt_folder = run_folder+"adapt/"
if not os.path.exists(adapt_folder):
    os.mkdir(adapt_folder)

#%%

if train == True:
    # Run the dataset script to generate the data
    # os.system(f'python dataset.py --split=adapt --savepath="{run_folder}" --seed="{seed}"')
    # os.system(f'python dataset.py --split=adapt_test --savepath="{run_folder}" --seed="{seed*2}"')
    os.system(f'python dataset.py --split=train --savepath="{run_folder}" --seed="{seed}"')
    os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}"')



#%%

print("=== Performing one per environment (OPE) ... === ")

adapt_mses = []

# for adapt_env in range(2):      ## Nb of adaptation environments
for adapt_env in [0, 8]:      ## Nb of adaptation environments
    print(f"\n\nCurrently training from scratch for environment: {adapt_env} \n")

    ## Define dataloader for training and validation
    # train_dataloader = DataLoader(run_folder+"adapt_data.npz", batch_size=-1, shuffle=True, key=seed)
    train_dataloader = DataLoader(run_folder+"train_data.npz", batch_size=-1, shuffle=True, key=seed)
    train_dataset = train_dataloader.dataset[adapt_env:adapt_env+1, ...]
    train_dataloader = DataLoader(dataset=train_dataset, t_eval=train_dataloader.t_eval, batch_size=-1, shuffle=True, key=seed)

    nb_envs = train_dataloader.nb_envs
    nb_trajs_per_env = train_dataloader.nb_trajs_per_env
    nb_steps_per_traj = train_dataloader.nb_steps_per_traj
    data_size = train_dataloader.data_size

    # val_dataloader = DataLoader(run_folder+"adapt_test_data.npz", shuffle=False)
    val_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)
    val_dataset = val_dataloader.dataset[adapt_env:adapt_env+1, ...]
    val_dataloader = DataLoader(dataset=val_dataset, t_eval=val_dataloader.t_eval, batch_size=-1, shuffle=False, key=seed)

    ##%%

    # print(val_dataloader.dataset.shape)
    plt.plot(val_dataloader.t_eval, val_dataloader.dataset[0, 0, :, 0])

    def circular_pad_2d(x, pad_width):
        """ Circular padding for 2D arrays """
        if isinstance(pad_width, int):
            pad_width = ((pad_width, pad_width), (pad_width, pad_width))
        # return jnp.pad(x, pad_width, mode='wrap')

        if x.ndim == 2:
            return jnp.pad(x, pad_width, mode='wrap')
        else:
            zero_pad = [(0,0)]*(x.ndim-2)
            return jnp.pad(x, zero_pad+list(pad_width), mode='wrap')
        # return jnp.pad(x, pad_width, mode='wrap')


    class Swish(eqx.Module):
        beta: jnp.ndarray
        def __init__(self, key=None):
            self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
        def __call__(self, x):
            return x * jax.nn.sigmoid(self.beta * x)

    class Augmentation(eqx.Module):
        layers_data: list
        layers_context: list
        layers_shared: list
        activations: list

        def __init__(self, data_res, kernel_size, nb_comp_chans, nb_hidden_chans, context_size, key=None):
            keys = generate_new_keys(key, num=12)
            circular_pad = lambda x: circular_pad_2d(x, kernel_size//2)
            self.activations = [Swish(key=keys[i]) for i in range(0, 6)]

            self.layers_context = [eqx.nn.Linear(context_size, data_res*data_res*2, key=keys[3]), self.activations[0],
                                    lambda x: jnp.stack(vec_to_mats(x, data_res, 2), axis=0),
                                circular_pad,
                                eqx.nn.Conv2d(2, nb_comp_chans, kernel_size, key=keys[0]), self.activations[5]]

            self.layers_data = [lambda x: jnp.stack(vec_to_mats(x, data_res, 2), axis=0),
                                circular_pad,
                                eqx.nn.Conv2d(2, nb_comp_chans, kernel_size, key=keys[0]), self.activations[4]]

            self.layers_shared = [circular_pad, 
                                eqx.nn.Conv2d(nb_comp_chans*1, nb_hidden_chans, kernel_size, key=keys[6]), self.activations[1],
                                circular_pad, 
                                eqx.nn.Conv2d(nb_hidden_chans, nb_hidden_chans, kernel_size, key=keys[7]), self.activations[2],
                                circular_pad, 
                                eqx.nn.Conv2d(nb_hidden_chans, nb_hidden_chans, kernel_size, key=keys[8]), self.activations[3],
                                circular_pad, 
                                eqx.nn.Conv2d(nb_hidden_chans, 2, kernel_size, key=keys[9]),
                                #   lambda x: x.flatten()]
                                lambda x: jnp.concatenate([x[0].flatten(), x[1].flatten()], axis=0)]

        def __call__(self, t, y, ctx):
            # return jnp.zeros_like(y)*ctx[0]

            for layer in self.layers_context:
                ctx = layer(ctx)

            for layer in self.layers_data:
                y = layer(y)

            # y = jnp.concatenate([y, ctx], axis=0)
            # y = jnp.concatenate([y, y], axis=0)
            for layer in self.layers_shared:
                y = layer(y)

            return y * 1e-2
            # return jnp.zeros_like(y)

    ## Second-order Taylor approximation
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

            # return vf(ctx)                                                                  ## TODO disable CSM

            # gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            # scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

            # print("all operand types:", type(vf), type(gradvf), type(scd_order_term))
            # return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term
            return vf(ctx_)


    augmentation = Augmentation(data_res=8, kernel_size=3, nb_comp_chans=8, nb_hidden_chans=64, context_size=context_size, key=seed)

    # physics = Physics(key=seed)
    physics = None

    vectorfield = ContextFlowVectorField(augmentation, physics=physics)
    print("\n\nTotal number of parameters in the model:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(vectorfield,eqx.is_array)) if x is not None), "\n\n")

    contexts = ContextParams(nb_envs, context_size, key=None)

    ## Define a custom loss function here
    def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s, key):

        # ind = jax.random.randint(key, shape=(context_pool_size,), minval=0, maxval=all_ctx_s.shape[0])
        # ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
        # ctx_s = all_ctx_s[ind, :]

        ## Select the two closes vectors in all_ctx_s to ctx
        dists = jnp.mean(jnp.abs(all_ctx_s-ctx), axis=1)
        ind = jnp.argsort(dists)[0:context_pool_size]        ## ctx itself, and the closest context
        ctx_s = all_ctx_s[ind, :]

        trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
        new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

        term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction
        # term2 = jnp.mean(ctx**2)             ## regularisation
        term2 = jnp.mean(jnp.abs(ctx))             ## regularisation
        term3 = params_norm_squared(model)

        loss_val = term1 + 1e-3*term2 + 1e-3*term3
        # loss_val = jnp.nan_to_num(term1, nan=0.0, posinf=0.0, neginf=0.0)
        # loss_val = term1 + 1e-3*term2

        return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)




    learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=seed)

    ## Define optimiser and traine the model

    nb_total_epochs = nb_epochs
    sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={nb_total_epochs//3:lr_factor, 2*nb_total_epochs//3:lr_factor})

    sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={nb_total_epochs//3:lr_factor, 2*nb_total_epochs//3:lr_factor})

    opt_node = optax.adabelief(sched_node)
    opt_ctx = optax.adabelief(sched_ctx)

    trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=seed)

    #%%

    trainer_save_path = run_folder if save_trainer == True else False
    if train == True:
        # for propostion in [0.25, 0.5, 0.75]:
        for i, prop in enumerate(np.linspace(1.0, 1.0, 1)):
            # trainer.dataloader.int_cutoff = int(prop*nb_steps_per_traj)
            trainer.train(nb_epochs=nb_epochs*(2**0), print_error_every=print_error_every*(2**0), update_context_every=0, save_path=trainer_save_path, key=seed, val_dataloader=val_dataloader, int_prop=prop)

    else:
        # print("\nNo training, attempting to load model and results from "+ run_folder +" folder ...\n")

        restore_folder = run_folder
        # restore_folder = "./runs/27012024-155719/finetune_193625/"
        trainer.restore_trainer(path=restore_folder)



    ##%%

    ## Test and visualise the results on a test dataloader

    visualtester = VisualTester(trainer)
    # ans = visualtester.trainer.nb_steps_node
    # print(ans.shape)

    ind_crit, _ = visualtester.test(val_dataloader, int_cutoff=1.0)

    visualtester.visualize(val_dataloader, int_cutoff=1.0, save_path=run_folder);

    adapt_mses.append(ind_crit)



#%%
print(adapt_mses)
print("Average MSE for adaptation:", np.mean(np.stack(adapt_mses)))
print("Std Dev MSE for adaptation:", np.std(np.stack(adapt_mses)))




















#%%
## If the nohup.log file exists, copy it to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        if adapt_test==True and train==False: ## Adaptation
            os.system(f"cp nohup.log {adapt_folder}")
        else:
            os.system(f"cp nohup.log {run_folder}")
            # os.system(f"open {run_folder}results_in_domain.png")
