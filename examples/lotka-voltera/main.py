from nodebias import *
from IPython.display import Image

# jax.config.update("jax_debug_nans", True)


#%%

# Image(filename="tmp/coda_lv_setting_1.png")
#%%
# Image(filename="tmp/coda_lv_setting_2.png")

## Additionally
# LV, GO: 4-layer MLPs with hidden layers of width 64
# We apply Swish activation (Ramachandran et al., 2018).
# The hypernet A is a single affine layer NN

#%%


## Hyperparams
SEED = 3
context_size = 20
nb_epochs = 1000



## Define dataloader for training

# raw_data = np.load("tmp/train_data.npz")
# dataset, t_eval = raw_data["X"][0:1, ...], raw_data["t"]
# train_dataloader = DataLoader(dataset, t_eval, batch_size=-1, int_cutoff=0.8, shuffle=True)

# train_dataloader = DataLoader("tmp/dataset_big.npz", batch_size=-1, int_cutoff=0.2, shuffle=True)
train_dataloader = DataLoader("tmp/train_data.npz", batch_size=-1, int_cutoff=0.25, shuffle=True)

nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

#%%

## Define model and loss function for the learner

activation = jax.nn.softplus
# activation = jax.nn.swish


# class Physics(eqx.Module):
#     params: jnp.ndarray

#     def __init__(self, key=None):
#         key = generate_new_keys(key, num=1)[0]
#         self.params = jax.random.uniform(key, (4,), minval=0.05, maxval=1.5)

#     def __call__(self, t, x, ctx):
#         dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
#         dx1 = x[0]*x[1]*self.params[3] - x[1]*self.params[2]
#         return jnp.array([dx0, dx1])

class Physics(eqx.Module):
    params: list
    layers_context: list

    def __init__(self, key=None):
        keys = generate_new_keys(key, num=12)
        # self.params = jax.random.uniform(keys[0], (4,), minval=0.05, maxval=1.5)
        self.params = [0, 0, 0, 0]

        width_size = 8
        self.layers_context = [eqx.nn.Linear(context_size, width_size*2, key=keys[0]), activation,
                        eqx.nn.Linear(width_size*2, width_size*2, key=keys[1]), activation,
                        eqx.nn.Linear(width_size*2, width_size, key=keys[2]), activation,
                        eqx.nn.Linear(width_size, 4, key=keys[3])]

    def __call__(self, t, x, ctx):
        # return jnp.zeros_like(x)
        # params = self.params
        params = ctx
        for layer in self.layers_context:
            params = layer(params) 

        params = jnp.abs(params)     ## TODO: this ensure posititity

        dx0 = x[0]*params[0] - x[0]*x[1]*params[1]
        dx1 = x[0]*x[1]*params[3] - x[1]*params[2]
        return jnp.array([dx0, dx1])

class Augmentation(eqx.Module):
    layers_data: list
    layers_context: list
    layers_shared: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=12)
        self.layers_data = [eqx.nn.Linear(data_size, width_size, key=keys[0]), activation,
                        # eqx.nn.Linear(width_size, width_size, key=keys[10]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

        self.layers_context = [eqx.nn.Linear(context_size, width_size, key=keys[3]), activation,
                        # eqx.nn.Linear(width_size, width_size, key=keys[11]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[4]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(data_size*2, width_size, key=keys[6]), activation,
                        # eqx.nn.Linear(width_size, width_size, key=keys[7]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[8]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[9])]


    def __call__(self, t, x, ctx):
        y = x
        ctx = ctx
        for i in range(len(self.layers_data)):
            y = self.layers_data[i](y)
            ctx = self.layers_context[i](ctx)

        # ctx = jnp.zeros_like(y)     ## TODO: remove this line, please !

        # y = jnp.concatenate([y, ctx, y*ctx], axis=0)
        y = jnp.concatenate([y, ctx], axis=0)
        # y = y*ctx
        for layer in self.layers_shared:
            y = layer(y)
        return y


# physics = Physics(key=SEED)
physics = None
augmentation = Augmentation(data_size=2, width_size=8*1, depth=3, context_size=context_size, key=SEED)
contexts = ContextParams(nb_envs, context_size, key=SEED)

# integrator = diffrax.Tsit5()
integrator = rk4_integrator

def loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, ctx_, key):
    # # trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx)
    # trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx, ctx_)
    # term1 = jnp.mean((trajs-trajs_hat)**2)
    # term2_1 = spectral_norm_estimation(model.vectorfield.neuralnet, key=key)
    # term2_2 = infinity_norm_estimation(model.vectorfield.neuralnet, trajs, ctx)
    # # term2_1 = 1e-2*jnp.mean((ctx)**2)
    # # term2_1 = 0.
    # # term2_2 = 0.
    # term2 = term2_1 + alpha*term2_2
    # loss_val = term1 + beta*term2
    # return loss_val, (jnp.sum(nb_steps), term1, term2)

    # ctx is singular, but ctx_ is plural, it is the contexts for all the environements
    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    # term1 = jnp.mean((new_trajs-trajs_hat)**2)

    weights = jnp.mean((jnp.broadcast_to(ctx, ctx_.shape)-ctx_)**2, axis=-1) + 1e-8
    weights = weights / jnp.sum(weights)
    term1 = jnp.mean((new_trajs-trajs_hat)**2, axis=(1,2,3))  ## TODO: give more weights to the points for this context itself. Introduce a weighting system
    term1 = jnp.sum(term1 * weights)

    term2 = 1e-1*jnp.mean((ctx)**2)
    loss_val = term1+term2       ### Dangerous, but limit the context TODO

    return loss_val, (jnp.sum(nb_steps)/ctx_.shape[0], term1, term2)


learner = Learner(augmentation, contexts, loss_fn_ctx, integrator, physics=physics, key=SEED)
# learner = Learner(augmentation, nb_envs, context_size, loss_fn_ctx, integrator, physics=None)



#%%

## Define optimiser and traine the model

nb_train_steps = nb_epochs * 2
# sched_node = optax.piecewise_constant_schedule(init_value=3e-3,
#                         boundaries_and_scales={int(nb_train_steps*0.25):0.2,
#                                                 int(nb_train_steps*0.5):0.01,
#                                                 int(nb_train_steps*0.75):0.05,
#                                                 int(nb_train_steps*0.9):0.2})
sched_node = 1e-3
# sched_node = optax.exponential_decay(3e-3, nb_epochs*2, 0.99)

# sched_ctx = optax.piecewise_constant_schedule(init_value=3e-2,
#                         boundaries_and_scales={int(nb_epochs*0.25):0.25,
#                                                 int(nb_epochs*0.5):0.25,
#                                                 int(nb_epochs*0.75):0.25})
sched_ctx = 1e-3

opt_node = optax.adabelief(sched_node)
opt_ctx = optax.adabelief(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=SEED)

#%%

# for propostion in [0.25, 0.5, 0.75]:
for propostion in np.linspace(0.25, 1.0, 2):
    trainer.dataloader.int_cutoff = int(propostion*nb_steps_per_traj)
    # nb_epochs = nb_epochs // 2 if nb_epochs > 1000 else 1000
    trainer.train(nb_epochs=nb_epochs, print_error_every=100, update_context_every=1, save_path="tmp/", key=SEED)

#%%

## Test and visualise the results on a test dataloader

# raw_data = np.load("tmp/test_data.npz")
# dataset, t_eval = raw_data["X"][0:1, ...], raw_data["t"]
# test_dataloader = DataLoader(dataset, t_eval)

test_dataloader = DataLoader("tmp/test_data.npz")

visualtester = VisualTester(test_dataloader, trainer)

print("Test score:", visualtester.test_cf(int_cutoff=1.0))

visualtester.visualise(int_cutoff=1.0, save_path="tmp/results.png");


#%%

## Print some other params
# print("Parameters in the physics component:", learner.neuralode.vectorfield.physics.params)


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
# train_dataloader.dataset[0,0].shape

# trainer.learner.physics.params
