#%%
### MAIN SCRIPT TO TRAIN A NEURAL CONTEXT FLOW ###

# %load_ext autoreload
# %autoreload 2

## Import all the necessary libraries
from ncf import *
from jax import random

## Seed for reproducibility in JAX
seed = 2026

## NCF main hyperparameters ##
context_pool_size = 3               ## Number of neighboring contexts j to use for a flow in env e
context_size = 202                 ## Size of the context vector

nb_outer_steps_max = 250           ## maximum number of outer steps when using NCF-T2
nb_inner_steps_max = 25             ## Maximum number of inner steps when using NCF-T2 (for both weights and contexts)
proximal_beta = 1e1                 ## Proximal coefficient, see beta in https://proceedings.mlr.press/v97/li19n.html
inner_tol_node = 1e-16               ## Tolerance for the inner optimisation on the weights
inner_tol_ctx = 1e-16                ## Tolerance for the inner optimisation on the contexts
early_stopping_patience = nb_outer_steps_max       ## Number of outer steps to wait before stopping early

## General training hyperparameters ##
print_error_every = 10              ## Print the error every n epochs
integrator = Euler                    ## Integrator to use for the learner
ivp_args = {"dt_init":1e-4, "rtol":1e-3, "atol":1e-6, "max_steps":4000, "subdivisions":1}
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


## Define a loss function for one environment (one context)
def loss_fn_env(model, trajs, t_eval, ctx, all_ctx_s, key):

    ## Define the context pool using the Random-All strategy
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)      ## reconstruction loss
    term2 = jnp.mean(jnp.abs(ctx))                  ## context regularisation
    term3 = params_norm_squared(model)              ## weight regularisation

    loss_val = term1 + 1e-3*term2 + 1e-3*term3

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)


## Create the neural network (accounting for the unknown in the system), and use physics is problem is known
neuralnet = FNO2D(modes1=8, 
                  modes2=8, 
                  width=10, 
                  depth=4, 
                  channels_last_proj=16, 
                  activation=jax.nn.softplus, 
                  out_channels=1, 
                  padding=0, 
                  key=jax.random.PRNGKey(seed), 
                  res=32, 
                  context_size=context_size)
vectorfield = SelfModulatedVectorField(physics=None, augmentation=neuralnet, taylor_order=taylor_order)
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
