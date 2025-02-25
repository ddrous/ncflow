import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'false'

# from collections import Callable
from ncf import *
# jax.config.update("jax_debug_nans", True)




#%%


seed = 2026
# seed = int(np.random.randint(0, 10000))

context_pool_size = 3               ## Number of neighboring contexts j to use for a flow in env e
context_size = 202
nb_epochs = 10000
nb_epochs_adapt = 1500
init_lr = 3e-4
lr_factor = 0.1

print_error_every = 5

train = True
# run_folder = "./runs/03082024-024220-Toy/"      ## Run folder to use when not training
reuse_run_folder = False

save_trainer = True

finetune = False

nb_outer_steps_max = 250
nb_inner_steps_max = 25
proximal_beta = 1e1
inner_tol_node = 1e-16
inner_tol_ctx = 1e-16
early_stopping_patience = nb_outer_steps_max//1

adapt = True
adapt_huge = False

# integrator = diffrax.Dopri5
integrator = Euler
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":4000, "subdivisions":1}
## subdivision is used for non-adaptive integrators like RK4. It's the number of extra steps to take between each evaluation time point

#%%


if train == True:

    # check that 'tmp' folder exists. If not, create it
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside 'tmp' whose name is the current time
    if reuse_run_folder == False:
        run_folder = './runs/'+time.strftime("%d%m%Y-%H%M%S")+'/'
        os.mkdir(run_folder)
    # run_folder = "./runs/23012024-163033/"
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

# os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}"')
# os.system(f'python dataset.py --split=adapt --savepath="{adapt_folder}" --seed="{seed*3}"');
# os.system(f'python dataset.py --split=adapt_test --savepath="{adapt_folder}" --seed="{seed*3}"');

if train == True and reuse_run_folder == False:
    # Run the dataset script to generate the data
    ## Copy data from tmp instead

    os.system(f'cp tmp/train_data.npz {run_folder}')
    os.system(f'cp tmp/test_data.npz {run_folder}')
if adapt == True and reuse_run_folder == False:
    os.system(f'cp tmp/adapt_data.npz {adapt_folder}')
    os.system(f'cp tmp/adapt_data_test.npz {adapt_folder}')
if adapt_huge == True and reuse_run_folder == False:
    # os.system(f'python dataset.py --split=adapt_huge --savepath="{adapt_folder}" --seed="{seed*4}"');
    # os.system(f'cp tmp/adapt_huge_data.npz {adapt_folder}')
    pass





#%%

## Define dataloader for training and validation
train_dataloader = DataLoader(run_folder+"train_data.npz", batch_size=-1, shuffle=True, key=seed)

nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

val_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)

#%%

## Define model and loss function for the learner

######## From https://github.com/astanziola/fourier-neural-operator-flax/blob/main/fno/modules.py

#%%

from jax import random

def normal(stddev=1e-2, dtype=jnp.float32):
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * stddev
    return init

class SpectralConv2d(eqx.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int

    kernel_1_r: jnp.ndarray
    kernel_1_i: jnp.ndarray
    kernel_2_r: jnp.ndarray
    kernel_2_i: jnp.ndarray

    def __init__(self, in_channels=1, out_channels=32, modes1=12, modes2=12, key=jax.random.PRNGKey(seed)):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        keys = jax.random.split(key, num=4)

        # Initializing the kernel parameters
        scale = 1 / (self.in_channels * self.out_channels)

        self.kernel_1_r = normal(scale)(keys[0], (self.in_channels, self.out_channels, self.modes1, self.modes2 ))
        self.kernel_1_i = normal(scale)(keys[1], (self.in_channels, self.out_channels, self.modes1, self.modes2))
        self.kernel_2_r = normal(scale)(keys[2], (self.in_channels, self.out_channels, self.modes1, self.modes2))
        self.kernel_2_i = normal(scale)(keys[3], (self.in_channels, self.out_channels, self.modes1, self.modes2))

    def __call__(self, x):
        # x.shape: [in_channels, height, width]
        x = x.transpose((1, 2, 0))[None,...]
        # NOW, x.shape: [batch, height, width, in_channels]

        # print("Just checking:", x.shape)

        height = x.shape[1]
        width = x.shape[2]

        # Checking that the modes are not more than the input size
        assert self.modes1 <= height // 2 + 1
        assert self.modes2 <= width // 2 + 1
        assert height % 2 == 0  # Only tested for even-sized inputs
        assert width % 2 == 0  # Only tested for even-sized inputs

        # Perform fft of the input
        x_ft = jnp.fft.rfftn(x, axes=(1, 2))

        # print("Channels in, out", self.in_channels, self.out_channels)
        # print("Shapes", x_ft.shape, self.kernel_1_r.shape, self.kernel_1_i.shape)

        # Multiply the center of the spectrum by the kernel
        out_ft = jnp.zeros_like(x_ft)
        s1 = jnp.einsum(
            'bijc,coij->bijo',
            x_ft[:, :self.modes1, :self.modes2, :],
            self.kernel_1_r + 1j * self.kernel_1_i)
        s2 = jnp.einsum(
            'bijc,coij->bijo',
            x_ft[:, -self.modes1:, :self.modes2, :],
            self.kernel_2_r + 1j * self.kernel_2_i)
        out_ft = out_ft.at[:, :self.modes1, :self.modes2, :].set(s1)
        out_ft = out_ft.at[:, -self.modes1:, :self.modes2, :].set(s2)

        # Go back to the spatial domain
        y = jnp.fft.irfftn(out_ft, axes=(1, 2))

        # print("Suceeded", y.shape)
        ## Remove the batch dimension
        y = y[0].transpose((2, 0, 1))

        return y

class FourierStage(eqx.Module):
    in_channels: int
    out_channels: int
    modes1: int
    modes2: int
    activation: callable

    spectral_conv: SpectralConv2d
    conv: eqx.nn.Conv

    def __init__(self, in_channels=32, out_channels=32, modes1=12, modes2=12, activation=jax.nn.softplus, key=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.activation = activation

        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.spectral_conv = SpectralConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes1=self.modes1,
            modes2=self.modes2,
            key=subkey1
        )
        self.conv = eqx.nn.Conv2d(
            self.in_channels,
            self.out_channels,
            (1, 1),
            key=subkey2
        )

    def __call__(self, x):
        x_fourier = self.spectral_conv(x)
        x_local = self.conv(x)
        return self.activation(x_fourier + x_local)


class FNO2D(eqx.Module):
    modes1: int
    modes2: int
    width: int
    depth: int
    channels_last_proj: int
    activation: callable
    out_channels: int
    padding: int

    res: int

    layers: list
    dense0: eqx.nn.Conv2d
    dense1: eqx.nn.Conv2d
    dense2: eqx.nn.Conv2d

    context_layer: eqx.nn.Linear

    def __init__(self, modes1=12, modes2=12, width=32, depth=4, channels_last_proj=32, 
                 activation=jax.nn.softplus, out_channels=1, padding=0, res=32, context_size=2, key=None):
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth
        self.channels_last_proj = channels_last_proj
        self.activation = activation
        self.out_channels = out_channels
        self.padding = padding

        self.res = res

        keys = jax.random.split(key, depth + 4)
        self.layers = [FourierStage(
            in_channels=self.width,
            out_channels=self.width,
            modes1=self.modes1,
            modes2=self.modes2,
            activation=self.activation if i < self.depth - 1 else lambda x: x,
            key=keys[i]
        ) for i in range(self.depth)]

        self.dense0 = eqx.nn.Conv2d(4, self.width, (1, 1), key=keys[-2])
        self.dense1 = eqx.nn.Conv2d(self.width, self.channels_last_proj, (1, 1), key=keys[-3])
        self.dense2 = eqx.nn.Conv2d(self.channels_last_proj, self.out_channels, (1, 1), key=keys[-4])

        self.context_layer = eqx.nn.Linear(context_size, self.res*self.res*1, key=keys[-1])

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x1, x2 = jnp.meshgrid(x1, x2, indexing='ij')
        grid = jnp.expand_dims(jnp.stack([x1, x2], axis=-1), 0)
        batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
        return batched_grid

    def __call__(self, t, x, ctx):

        ## Batch size always 1
        x = x.reshape((self.res, self.res, -1))

        ## Process context
        ctx = self.context_layer(ctx).reshape((self.res, self.res, 1))

        # Generate coordinate grid, and append to input channels
        grid = self.get_grid(x).squeeze()
        # x = jnp.concatenate([x, grid], axis=-1)
        x = jnp.concatenate([x, ctx, grid], axis=-1)

        ## Put channel first
        x = x.transpose((2, 0, 1))

        # Lift the input to a higher dimension
        x = self.dense0(x)

        # Pad input
        if self.padding > 0:
            x = jnp.pad(
                x,
                ((0, 0), (0, self.padding), (0, self.padding), (0, 0)),
                mode='constant'
            )

        # Apply Fourier stages
        for layer in self.layers:
            x = layer(x)

        # Unpad
        if self.padding > 0:
            x = x[:-self.padding, :-self.padding, :]

        # Project to the output channels
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)

        ## Reshape for next step
        # x = x.reshape((1, self.res, self.res, self.out_channels))
        x = x.reshape((self.res*self.res*self.out_channels,))

        return x














# # First-order Taylor approximation
# class ContextFlowVectorField(eqx.Module):
#     physics: eqx.Module
#     augmentation: eqx.Module

#     def __init__(self, augmentation, physics=None):
#         self.augmentation = augmentation
#         self.physics = physics

#     def __call__(self, t, x, ctxs):
#         if self.physics is None:
#             vf = lambda xi_: self.augmentation(t, x, xi_)
#         else:
#             vf = lambda xi_: self.physics(t, x, xi_) + self.augmentation(t, x, xi_)

#         gradvf = lambda xi_, xi: eqx.filter_jvp(vf, (xi_,), (xi-xi_,))[1]

#         ctx, ctx_ = ctxs
#         return vf(ctx_) + gradvf(ctx_, ctx)


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

        gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
        scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]

        # print("all operand types:", type(vf), type(gradvf), type(scd_order_term))
        return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term


# augmentation = Augmentation(data_res=8, kernel_size=3, nb_comp_chans=8, nb_hidden_chans=64, context_size=context_size, key=seed)
augmentation = FNO2D(modes1=8, modes2=8, width=10, depth=4, channels_last_proj=16, activation=jax.nn.softplus, out_channels=1, padding=0, key=jax.random.PRNGKey(seed), res=32, context_size=context_size)

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


#%%

## Define optimiser and traine the model

nb_total_epochs = nb_outer_steps_max * nb_inner_steps_max
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
        # trainer.train(nb_epochs=nb_epochs*(2**0), print_error_every=print_error_every*(2**0), update_context_every=1, save_path=trainer_save_path, key=seed, val_dataloader=val_dataloader, int_prop=prop)
        trainer.train_proximal(nb_outer_steps_max=nb_outer_steps_max, 
                               nb_inner_steps_max=nb_inner_steps_max, 
                               proximal_reg=proximal_beta, 
                               inner_tol_node=inner_tol_node, 
                               inner_tol_ctx=inner_tol_ctx,
                               print_error_every=print_error_every*(2**0), 
                               save_path=trainer_save_path, 
                               val_dataloader=val_dataloader, 
                               patience=early_stopping_patience,
                               int_prop=prop,
                               key=seed)
else:
    # print("\nNo training, attempting to load model and results from "+ run_folder +" folder ...\n")

    restore_folder = run_folder
    # restore_folder = "./runs/27012024-155719/finetune_193625/"
    trainer.restore_trainer(path=restore_folder)



#%%
















if finetune:
    # ## Finetune a trained model

    finetunedir = run_folder+"finetune_"+trainer.dataloader.data_id+"/"
    if not os.path.exists(finetunedir):
        os.mkdir(finetunedir)
    print("No training. Loading anctx_sd finetuning into:", finetunedir)

    trainer.dataloader.int_cutoff = nb_steps_per_traj

    opt_node = optax.adabelief(1e-7)
    opt_ctx = optax.adabelief(1e-7)
    trainer.opt_node, trainer.opt_ctx = opt_node, opt_ctx

    # trainer.opt_node_state = trainer.opt_node.init(eqx.filter(trainer.learner.neuralode, eqx.is_array))
    # trainer.opt_ctx_state = trainer.opt_ctx.init(trainer.learner.contexts)

    trainer.train(nb_epochs=24000, print_error_every=1000, update_context_every=1, save_path=finetunedir, key=seed)





#%%

## Test and visualise the results on a test dataloader

test_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False)
visualtester = VisualTester(trainer)
# ans = visualtester.trainer.nb_steps_node
# print(ans.shape)

ind_crit = visualtester.test(test_dataloader, int_cutoff=1.0)
print("\nPer-environment IND scores:", ind_crit[1])

if finetune:
    savefigdir = finetunedir+"results_in_domain.png"
else:
    savefigdir = run_folder+"results_in_domain.png"
visualtester.visualize(test_dataloader, int_cutoff=1.0, save_path=savefigdir);



#%%

## Custom Gray-Scott trajectory visualiser
if finetune:
    savefigdir = finetunedir+"results_2D_ind.png"
else:
    savefigdir = run_folder+"results_2D_ind.png"
# visualtester.visualize2D(test_dataloader, int_cutoff=1.0, res=8, nb_plot_timesteps=10, save_path=savefigdir);
# visualtester.visualize2D(train_dataloader, int_cutoff=1.0, res=8, nb_plot_timesteps=10, save_path=run_folder+"results_2D_ind_train.png");


#%%
# len(trainer.losses_node

# ## Run and get the contexts
# for i in range(nb_envs):
#     ctx = trainer.learner.contexts.params[i]
#     # print(ctx)
#     param = ctx
#     for layer in trainer.learner.physics.layers_context:
#         param = layer(param)
#         # print("Context", ctx, "     Param", param)
#     param = jnp.abs(param)
#     print("Param:", param)




#%%
## If the nohup.log file exists, copy it to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        if finetune == True:
            os.system(f"cp nohup.log {finetunedir}")
            ## Open the results_in_domain in the terminal
            os.system(f"open {finetunedir}results_in_domain.png")
        else:
            os.system(f"cp nohup.log {run_folder}")
            # os.system(f"open {run_folder}results_in_domain.png")



#%%

























## Give the dataloader an id to help with restoration later on

adapt_dataloader = DataLoader(adapt_folder+"adapt_data.npz", adaptation=True, data_id="170846", key=seed)
adapt_dataloader_test = DataLoader(adapt_folder+"adapt_data_test.npz", adaptation=True, data_id="1708460", key=seed)

# sched_ctx_new = optax.piecewise_constant_schedule(init_value=1e-5,
#                         boundaries_and_scales={int(nb_epochs_adapt*0.25):1.,
#                                                 int(nb_epochs_adapt*0.5):0.1,
#                                                 int(nb_epochs_adapt*0.75):1.})
# sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr,
#                         boundaries_and_scales={nb_total_epochs//3:0.1, 2*nb_total_epochs//3:1.0})
sched_ctx_new = 5e-4
opt_adapt = optax.adabelief(sched_ctx_new)

if adapt == True:
    trainer.adapt_sequential(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every**2, save_path=adapt_folder)
    # trainer.adapt(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every**2, save_path=adapt_folder)
else:
    print("save_id:", adapt_dataloader.data_id)

    trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)


#%%
ood_crit = visualtester.test(adapt_dataloader_test, int_cutoff=1.0)      ## It's the same visualtester as before during training. It knows trainer

visualtester.visualize(adapt_dataloader, int_cutoff=1.0, save_path=adapt_folder+"results_ood.png");


print("\nPer-environment OOD scores:", ood_crit[1])



#%%
## If the nohup.log file exists, copy it to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        if finetune == True:
            os.system(f"cp nohup.log {finetunedir}")
            ## Open the results_in_domain in the terminal
            os.system(f"open {finetunedir}results_in_domain.png")
        else:
            os.system(f"cp nohup.log {run_folder}")
            # os.system(f"open {run_folder}results_in_domain.png")


#%%

# eqx.tree_deserialise_leaves(run_folder+"contexts.eqx", learner.contexts)















#%%

#### Generate data for analysis


# ## We want to store 3 values in a CSV file: "seed", "ind_crit", and "ood_crit", into the test_scores.csv file


# print("\nFull evaluation of the model on 10 random seeds\n", flush=True)

# # First, check if the file exists. If not, create it and write the header
# if not os.path.exists(run_folder+'analysis'):
#     os.mkdir(run_folder+'analysis')

# csv_file = run_folder+'analysis/test_scores.csv'
# if not os.path.exists(csv_file):
#     os.system(f"touch {csv_file}")

# with open(csv_file, 'r') as f:
#     lines = f.readlines()
#     if len(lines) == 0:
#         with open(csv_file, 'w') as f:
#             f.write("seed,ind_crit,ood_crit\n")


# ## Get results on test and adaptation datasets, then append them to the csv

# np.random.seed(seed)
# seeds = np.random.randint(0, 10000, 10)
# for seed in seeds:
# # for seed in range(8000, 6*10**3, 10):
#     os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}" --verbose=0')
#     os.system(f'python dataset.py --split=adapt --savepath="{adapt_folder}" --seed="{seed*3}" --verbose=0')

#     test_dataloader = DataLoader(run_folder+"test_data.npz", shuffle=False, batch_size=1, data_id="082026")
#     adapt_test_dataloader = DataLoader(adapt_folder+"adapt_data.npz", adaptation=True, batch_size=1, key=seed, data_id="082026")

#     ind_crit, _ = visualtester.test(test_dataloader, int_cutoff=1.0, verbose=False)
#     ood_crit, _ = visualtester.test(adapt_test_dataloader, int_cutoff=1.0, verbose=False)

#     with open(csv_file, 'a') as f:
#         f.write(f"{seed},{ind_crit},{ood_crit}\n")


# ## Print the mean and stds of the scores
# import pandas as pd
# pd.set_option('display.float_format', '{:.2e}'.format)
# test_scores = pd.read_csv(csv_file).describe()
# print(test_scores.iloc[:3])

