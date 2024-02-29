# import os
## Do not preallocate GPU memory
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = '\"platform\"'

from nodax import *
# jax.config.update("jax_debug_nans", True)

## Execute jax on CPU
# jax.config.update("jax_platform_name", "cpu")





#%%

seed = 2026
# seed = int(np.random.randint(0, 10000))

context_pool_size = 2               ## Number of neighboring contexts j to use for a flow in env e
context_size = 256//2
nb_epochs = 1500
nb_epochs_adapt = 1000
init_lr = 1e-3
sched_factor = 0.5            ## Multiply the lr by this factor at each third of the training

print_error_every = 100

train = True
run_folder = "./runs/27022024-104335/"      ## Run folder to use when not training

save_trainer = True

finetune = False

adapt_test = False
adapt_restore = False


activation = jax.nn.softplus
# activation = jax.nn.swish
# activation = lambda x:x
# activation = jax.nn.sigmoid


# integrator = diffrax.Dopri5
integrator = RK4
ivp_args = {"dt_init":1e-7, "rtol":1e-2, "atol":1e-4, "max_steps":40000, "subdivisions":500}
## subdivision is used for non-adaptive integrators like RK4. It's the number of extra steps to take between each evaluation time point

#%%


if train == True:

    # check that 'tmp' folder exists. If not, create it
    if not os.path.exists('./runs'):
        os.mkdir('./runs')

    # Make a new folder inside 'tmp' whose name is the current time
    run_folder = './runs/'+time.strftime("%d%m%Y-%H%M%S")+'/'
    # run_folder = "./runs/23012024-163033/"
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
    os.system(f'python dataset.py --split=train --savepath="{run_folder}" --seed="{seed}"')
    os.system(f'python dataset.py --split=test --savepath="{run_folder}" --seed="{seed*2}"')




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


class My2DConv(eqx.Module):
    """ A conbcolution meant to approximate a differential operator : https://arxiv.org/abs/2402.16845"""
    in_channels: int
    out_channels: int
    kernel_size: int
    weight: jnp.ndarray

    def __init__(self, in_channels, out_channels, kernel_size, use_bias=False, key=None):
        keys = generate_new_keys(key, num=2)

        kernel_size = (kernel_size, kernel_size)
        lim = 1 / np.sqrt(in_channels * np.prod(kernel_size))
        self.weight = jax.random.uniform(keys[0], (out_channels, in_channels) + kernel_size, minval=-lim, maxval=lim)
        # if use_bias:
        #     self.bias = jax.random.uniform(keys[1], (out_channels,) + (1,) * 2, minval=-lim, maxval=lim,)
        # else:
        #     self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def __call__(self, x):
        x = circular_pad(x[None,...])
        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.weight - jnp.mean(self.weight),        ## trick to obtain diff operator
            window_strides=(1,1),
            padding="VALID",
            rhs_dilation=(1,1),
            feature_group_count=1,
        )
        x = jnp.squeeze(x, axis=0)
        # if self.use_bias:
        #     x = x + self.bias
        return x


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
    activation: Swish


    def __init__(self, data_res, kernel_size, nb_int_channels, context_size, key=None):

        chans = 16

        keys = generate_new_keys(key, num=12)
        circular_pad = lambda x: circular_pad_2d(x, kernel_size//2)
        # activation = self.activation = Swish(key=keys[10])
        activation = self.activation = jax.nn.sigmoid

        self.layers_context = [eqx.nn.Linear(context_size, data_res*data_res*nb_int_channels, key=keys[3]), activation,
                                lambda x: jnp.stack(vec_to_mats(x, data_res, nb_int_channels), axis=0)]

        self.layers_data = [lambda x: jnp.stack(vec_to_mats(x, data_res, 2), axis=0)]
                            # circular_pad,
                            # eqx.nn.Conv2d(2, nb_int_channels, kernel_size, key=keys[0]), activation]

        self.layers_shared = [
            # circular_pad, 
                              My2DConv(4, chans, kernel_size, key=keys[6]), activation,
                            #   circular_pad, 
                            #   eqx.nn.Conv2d(chans, chans, kernel_size, key=keys[7]), activation,
                            #   circular_pad, 
                              My2DConv(chans, chans, kernel_size, key=keys[8]), activation,
                            #   circular_pad, 
                              My2DConv(chans, 2, kernel_size, key=keys[9]), activation,
                              lambda x: x.flatten()]


        # ## Attempt with MLP
        # self.layers_shared = [eqx.nn.Linear(32*32*2, 128, key=keys[6]), activation,
        #                         eqx.nn.Linear(128, 128, key=keys[7]), activation,
        #                         eqx.nn.Linear(128, 32*32*2, key=keys[8])]

    def __call__(self, t, y, ctx):
        # return jnp.zeros_like(y)*ctx[0]

        for layer in self.layers_context:
            ctx = layer(ctx)

        for layer in self.layers_data:
            y = layer(y)

        # # y = jnp.concatenate([y, ctx], axis=0)
        y = jnp.concatenate([y, y], axis=0)
        # y_ = y
        for layer in self.layers_shared:
            y = layer(y)

        return y
        # return jnp.zeros_like(y)



def vec_to_mat(vec_uv, res=32):
    UV = jnp.split(vec_uv, 2)
    U = jnp.reshape(UV[0], (res, res))
    V = jnp.reshape(UV[1], (res, res))
    return U, V

def mat_to_vec(mat_U, mat_V, res):
    dudt = jnp.reshape(mat_U, res * res)
    dvdt = jnp.reshape(mat_V, res * res)
    return jnp.concatenate((dudt, dvdt))

def laplacian2D(a):
    # a_nn | a_nz | a_np
    # a_zn | a    | a_zp
    # a_pn | a_pz | a_pp
    a_zz = a

    a_nz = jnp.roll(a_zz, (+1, 0), (0, 1))
    a_pz = jnp.roll(a_zz, (-1, 0), (0, 1))
    a_zn = jnp.roll(a_zz, (0, +1), (0, 1))
    a_zp = jnp.roll(a_zz, (0, -1), (0, 1))

    a_nn = jnp.roll(a_zz, (+1, +1), (0, 1))
    a_np = jnp.roll(a_zz, (+1, -1), (0, 1))
    a_pn = jnp.roll(a_zz, (-1, +1), (0, 1))
    a_pp = jnp.roll(a_zz, (-1, -1), (0, 1))

    return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (1. ** 2)

# class Physics(eqx.Module):
#     layers: list
#     # number: jnp.ndarray

#     def __init__(self, key=None):
#         keys = generate_new_keys(key, num=4)
#         width_size = 8
#         # # new_act = jax.nn.sigmoid
#         self.layers = [eqx.nn.Linear(context_size, width_size*2, key=keys[0]), activation,
#                         eqx.nn.Linear(width_size*2, width_size*2, key=keys[1]), activation,
#                         eqx.nn.Linear(width_size*2, width_size*2, key=keys[2]), activation,
#                         eqx.nn.Linear(width_size*2, 4, key=keys[3])]
#         # self.number = jax.random.uniform(keys[0], shape=(1,), minval=0.01, maxval=0.5)

#     def __call__(self, t, uv, ctx):
#         params = ctx
#         for layer in self.layers:
#             params = layer(params)
#         params = jnp.abs(params)
#         # params = jnp.array([0.2097, 0.105, 0.03, 0.062])
#         # params = jnp.array([0.2097, 0.105, 0.03, self.number[0]*ctx[0]])

#         U, V = vec_to_mat(uv, 32)
#         deltaU = laplacian2D(U)
#         deltaV = laplacian2D(V)
#         dUdt = (params[0] * deltaU - U * (V ** 2) + params[2] * (1. - U))
#         dVdt = (params[1] * deltaV + U * (V ** 2) - (params[2] + params[3]) * V)
#         duvdt = mat_to_vec(dUdt, dVdt, 32)

#         duvdt = jnp.nan_to_num(duvdt, nan=0.0, posinf=0.0, neginf=0.0)
#         return duvdt


# init_lr = 1e-6
circular_pad = lambda x: circular_pad_2d(x, 3//2)
# new_act = lambda x: jnp.clip(x, a_min=0.0, a_max=1.0)
# new_act = jax.nn.sigmoid
new_act = jax.nn.swish


class Physics(eqx.Module):
    layers: list
    # number: jnp.ndarray

    def __init__(self, key=None):
        keys = generate_new_keys(key, num=6)
        width_size = 8
        # # new_act = jax.nn.sigmoid
        # self.layers = [eqx.nn.Conv2d(1, 1, 3, use_bias=False, key=keys[1]),
        #                 eqx.nn.Conv2d(1, 1, 3, use_bias=False, key=keys[2])]
        # true_kernel = jnp.array([[[[0.25, 0.5, 0.25], [0.5, -3., 0.5], [0.25, 0.5, 0.25]]]])
        # guess_kernel1 = true_kernel + 1e-2*jax.random.normal(keys[0], shape=(1, 1, 3, 3))
        # guess_kernel2 = true_kernel + 1e-2*jax.random.normal(keys[1], shape=(1, 1, 3, 3))
        # self.layers[0] = eqx.tree_at(lambda l:l.weight, self.layers[0], guess_kernel1)
        # self.layers[1] = eqx.tree_at(lambda l:l.weight, self.layers[1], guess_kernel2)

        # pad = 0
        # self.layers = [eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=pad, key=keys[1]),
        #                 # eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=1, key=keys[2]),
        #                 # eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=1, key=keys[3]),
        #                 # eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=1, key=keys[4]),
        #                 eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=pad, key=keys[5]),
        #                 eqx.nn.Conv2d(1, 1, 3, use_bias=False, padding=pad, key=keys[0])]

        pad = 0
        self.layers = [My2DConv(1, 1, 3, use_bias=False, key=keys[0]),
                        My2DConv(1, 1, 3, use_bias=False, key=keys[1])]



    def __call__(self, t, uv, ctx):
        # params = ctx
        # for layer in self.layers:
        #     params = layer(params)
        # params = jnp.abs(params)
        params = jnp.array([0.2097, 0.105, 0.03, 0.062])
        # params = jnp.array([0.2097, 0.105, 0.03, self.number[0]*ctx[0]])

        U, V = vec_to_mat(uv, 32)
        # deltaU = self.layers[2](circular_pad(self.layers[1](circular_pad(self.layers[0](circular_pad(U[None,...]))))))[0,...]
        # deltaV = self.layers[5](circular_pad(self.layers[4](circular_pad(self.layers[3](circular_pad(V[None,...]))))))[0,...]
        # deltaU = activation(deltaU)
        # deltaV = activation(deltaV)

        # deltaU = U[None,...]
        # deltaV = V[None,...]
        # for i in range(3):
        #     deltaU = circular_pad(deltaU)
        #     deltaV = circular_pad(deltaV)
        #     deltaU = self.layers[i](deltaU)
        #     deltaV = self.layers[i](deltaV)
        # deltaU = new_act(deltaU)
        # deltaV = new_act(deltaV)
        # deltaU = deltaU[0,...]
        # deltaV = deltaV[0,...]

        deltaU = U[None,...]
        deltaV = V[None,...]
        for i in range(2):
            deltaU = self.layers[i](deltaU)
            deltaV = self.layers[i](deltaV)

        # deltaV = self.layers[1](circular_pad(V[None,...]))[0,...]
        # deltaU = laplacian2D(U)
        # deltaV = laplacian2D(V)
        dUdt = (params[0] * deltaU - U * (V ** 2) + params[2] * (1. - U))
        dVdt = (params[1] * deltaV + U * (V ** 2) - (params[2] + params[3]) * V)
        duvdt = mat_to_vec(dUdt, dVdt, 32)

        # duvdt = jnp.nan_to_num(duvdt, nan=0.0, posinf=0.0, neginf=0.0)
        return duvdt




class ContextFlowVectorField(eqx.Module):
    physics: eqx.Module
    augmentation: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics

    def __call__(self, t, x, ctxs):
        if self.physics is None:
            vf = lambda xi_: self.augmentation(t, x, xi_)
        else:
            # vf = lambda xi_: self.physics(t, x, xi_) + self.augmentation(t, x, xi_)
            vf = lambda xi_: self.physics(t, x, xi_)

        gradvf = lambda xi_, xi: eqx.filter_jvp(vf, (xi_,), (xi-xi_,))[1]

        ctx, ctx_ = ctxs
        # return vf(ctx_) + gradvf(ctx_, ctx)
        return vf(ctx)


augmentation = Augmentation(data_res=32, kernel_size=3, nb_int_channels=4, context_size=context_size, key=seed)

# physics = Physics(key=seed)
physics = None

vectorfield = ContextFlowVectorField(augmentation, physics=physics)
print("\n\nTotal number of parameters in the model:", sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(vectorfield,eqx.is_array)) if x is not None), "\n\n")

contexts = ContextParams(nb_envs, context_size, key=None)

## Define a custom loss function here
def loss_fn_ctx(model, trajs, t_eval, ctx, all_ctx_s, key):

    # ind = jax.random.randint(key, shape=(context_pool_size,), minval=0, maxval=all_ctx_s.shape[0])
    ind = jax.random.permutation(key, all_ctx_s.shape[0])[:context_pool_size]
    ctx_s = all_ctx_s[ind, :]

    # jax.debug.print("indices chosen for this loss {}", ind)

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_s)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)  ## reconstruction
    term2 = jnp.mean(ctx**2)             ## regularisation

    # loss_val = term1 + 1e-3*term2
    # loss_val = jnp.nan_to_num(term1, nan=0.0, posinf=0.0, neginf=0.0)
    loss_val = term1

    return loss_val, (jnp.sum(nb_steps)/ctx_s.shape[0], term1, term2)




learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=seed)


#%%

## Define optimiser and traine the model

nb_total_epochs = nb_epochs * 1
sched_node = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})

sched_ctx = optax.piecewise_constant_schedule(init_value=init_lr,
                        boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})

opt_node = optax.adam(sched_node)
opt_ctx = optax.adam(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=seed)

#%%

# ## Print the dataset
# print("Dataset shape now:", train_dataloader.dataset)
# print()

trainer_save_path = run_folder if save_trainer == True else False
if train == True:
    # for i, prop in enumerate(np.linspace(0.25, 1.0, 3)):
    for i, prop in enumerate(np.linspace(1.0, 1.0, 1)):
        # trainer.dataloader.int_cutoff = int(prop*nb_steps_per_traj)
        trainer.train(nb_epochs=nb_epochs*(2**0), print_error_every=print_error_every*(2**0), update_context_every=1, save_path=trainer_save_path, key=seed, val_dataloader=val_dataloader, int_prop=prop)

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
visualtester.visualize2D(test_dataloader, int_cutoff=1.0, res=32, save_path=savefigdir);


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

























## Give the dataloader an id to help with restoration later on

if adapt_test and not adapt_restore:
    os.system(f'python dataset.py --split=adapt --savepath="{adapt_folder}" --seed="{seed*3}"');

if adapt_test:
    adapt_dataloader = DataLoader(adapt_folder+"adapt_data.npz", adaptation=True, data_id="170846", key=seed)

    # sched_ctx_new = optax.piecewise_constant_schedule(init_value=1e-5,
    #                         boundaries_and_scales={int(nb_epochs_adapt*0.25):1.,
    #                                                 int(nb_epochs_adapt*0.5):0.1,
    #                                                 int(nb_epochs_adapt*0.75):1.})
    sched_ctx_new = optax.piecewise_constant_schedule(init_value=init_lr,
                            boundaries_and_scales={nb_total_epochs//3:sched_factor, 2*nb_total_epochs//3:sched_factor})
    # sched_ctx_new = 1e-5
    opt_adapt = optax.adabelief(sched_ctx_new)

    if adapt_restore == False:
        trainer.adapt(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every, save_path=adapt_folder)
    else:
        print("Save_id for restoring trained adapation model:", adapt_dataloader.data_id)
        trainer.restore_adapted_trainer(path=adapt_folder, data_loader=adapt_dataloader)


#%%
if adapt_test:
    ood_crit = visualtester.test(adapt_dataloader, int_cutoff=1.0)      ## It's the same visualtester as before during training. It knows trainer

    visualtester.visualize(adapt_dataloader, int_cutoff=1.0, save_path=adapt_folder+"results_ood.png");




#%%
## If the nohup.log file exists, copy it to the run folder
try:
    __IPYTHON__ ## in a jupyter notebook
except NameError:
    if os.path.exists("nohup.log"):
        if finetune == True:
            os.system(f"cp nohup.log {finetunedir}")
            ## Open the results_in_domain in the terminal
            # os.system(f"open {finetunedir}results_in_domain.png")
        else:
            os.system(f"cp nohup.log {run_folder}")
            # os.system(f"open {run_folder}results_in_domain.png")


#%%

# # eqx.tree_deserialise_leaves(run_folder+"contexts.eqx", learner.contexts)
# print("Kernel layer 1\n", trainer.learner.neuralode.vectorfield.physics.layers[0].weight)
# print("Kernel layer 2\n", trainer.learner.neuralode.vectorfield.physics.layers[1].weight)














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

