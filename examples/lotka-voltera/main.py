from nodax import *

# jax.config.update("jax_debug_nans", True)


#%%


## Hyperparams
SEED = 18
context_size = 8000
nb_epochs = 5000
nb_epochs_adapt = 5000

print_error_every = 1000


## Define dataloader for training

# raw_data = np.load("tmp/train_data.npz")
# dataset, t_eval = raw_data["X"][0:1, ...], raw_data["t"]
# train_dataloader = DataLoader(dataset, t_eval, batch_size=-1, int_cutoff=0.8, shuffle=True)

# train_dataloader = DataLoader("tmp/dataset_big.npz", batch_size=-1, int_cutoff=0.2, shuffle=True)
train_dataloader = DataLoader("tmp/train_data.npz", batch_size=-1, int_cutoff=0.25, shuffle=True, key=SEED)

nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

#%%

## Define model and loss function for the learner

activation = jax.nn.softplus
# activation = jax.nn.swish

class Physics(eqx.Module):
    layers: list

    def __init__(self, width_size=8, key=None):
        keys = generate_new_keys(key, num=4)
        self.layers = [eqx.nn.Linear(context_size, width_size*2, key=keys[0]), activation,
                        eqx.nn.Linear(width_size*2, width_size*2, key=keys[1]), activation,
                        eqx.nn.Linear(width_size*2, width_size, key=keys[2]), activation,
                        eqx.nn.Linear(width_size, 4, key=keys[3])]

    def __call__(self, t, x, ctx):
        params = ctx
        for layer in self.layers:
            params = layer(params)
        params = jnp.abs(params)

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
                        eqx.nn.Linear(width_size, width_size, key=keys[10]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

        self.layers_context = [eqx.nn.Linear(context_size, context_size//10, key=keys[3]), activation,
                        eqx.nn.Linear(context_size//10, width_size*4, key=keys[11]), activation,
                        eqx.nn.Linear(width_size*4, width_size, key=keys[4]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(data_size*2, width_size, key=keys[6]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[7]), activation,
                        eqx.nn.Linear(width_size, width_size, key=keys[8]), activation,
                        eqx.nn.Linear(width_size, data_size, key=keys[9])]


    def __call__(self, t, x, ctx):
        y = x
        ctx = ctx
        for i in range(len(self.layers_data)):
            y = self.layers_data[i](y)
            ctx = self.layers_context[i](ctx)

        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)
        return y

class ContextFlowVectorField(eqx.Module):
    physics: eqx.Module
    augmentation: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics if physics is not None else NoPhysics()

    def __call__(self, t, x, ctx, ctx_):

        vf = lambda xi_: self.physics(t, x, xi_) + self.augmentation(t, x, xi_)
        gradvf = lambda xi_, xi: eqx.filter_jvp(vf, (xi_,), (xi-xi_,))[1]

        return vf(ctx_) + gradvf(ctx_, ctx)
        # return vf(ctx)



# physics = Physics(key=SEED)
physics = None
augmentation = Augmentation(data_size=2, width_size=8*4, depth=3, context_size=context_size, key=SEED)
vectorfield = ContextFlowVectorField(augmentation, physics=physics)

# contexts = ContextParams(nb_envs, context_size, key=SEED)
contexts = ContextParams(nb_envs, context_size, key=None)

# integrator = diffrax.Tsit5()
integrator = rk4_integrator


# loss_fn_ctx = basic_loss_fn_ctx
# loss_fn_ctx = default_loss_fn_ctx

## Redefine a proper loss function here
def loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, ctx_, key):

    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)

    term2 = 1e-1*jnp.mean((ctx)**2)

    loss_val = term1+term2

    return loss_val, (jnp.sum(nb_steps)/ctx_.shape[0], term1, term2)


learner = Learner(vectorfield, contexts, loss_fn_ctx, integrator, key=SEED)
# learner = Learner(augmentation, nb_envs, context_size, loss_fn_ctx, integrator, physics=None)



#%%

## Define optimiser and traine the model

nb_train_steps = nb_epochs * 3
sched_node = optax.piecewise_constant_schedule(init_value=3e-4,
                        boundaries_and_scales={int(nb_train_steps*0.25):0.2,
                                                int(nb_train_steps*0.5):0.1,
                                                int(nb_train_steps*0.75):0.01})
# sched_node = 1e-3
# sched_node = optax.exponential_decay(3e-3, nb_epochs*2, 0.99)

sched_ctx = optax.piecewise_constant_schedule(init_value=3e-3,
                        boundaries_and_scales={int(nb_epochs*0.25):0.2,
                                                int(nb_epochs*0.5):0.1,
                                                int(nb_epochs*0.75):0.01})
# sched_ctx = 1e-3

opt_node = optax.adabelief(sched_node)
opt_ctx = optax.adabelief(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=SEED)

#%%

# for propostion in [0.25, 0.5, 0.75]:
for i, prop in enumerate(np.linspace(0.25, 1.0, 2)):
    trainer.dataloader.int_cutoff = int(prop*nb_steps_per_traj)
    # nb_epochs = nb_epochs // 2 if nb_epochs > 1000 else 1000
    trainer.train(nb_epochs=nb_epochs*(2**i), print_error_every=print_error_every*(2**i), update_context_every=1, save_path="tmp/", key=SEED)

#%%

## Test and visualise the results on a test dataloader

# raw_data = np.load("tmp/test_data.npz")
# dataset, t_eval = raw_data["X"][0:1, ...], raw_data["t"]
# test_dataloader = DataLoader(dataset, t_eval)

test_dataloader = DataLoader("tmp/test_data.npz", shuffle=False)

visualtester = VisualTester(trainer)

ind_crit = visualtester.test(test_dataloader, int_cutoff=1.0)

visualtester.visualize(test_dataloader, int_cutoff=1.0, save_path="tmp/results.png");


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

adapt_dataloader = DataLoader("tmp/ood_data.npz", adaptation=True, key=SEED)

opt_adapt = optax.adabelief(default_optimizer_schedule(3e-3, nb_epochs_adapt))

trainer.adapt(adapt_dataloader, nb_epochs=nb_epochs_adapt, optimizer=opt_adapt, print_error_every=print_error_every, save_path="tmp/")


#%%
ood_crit = visualtester.test(adapt_dataloader, int_cutoff=1.0)

visualtester.visualize(adapt_dataloader, int_cutoff=1.0, save_path="tmp/results_adapt.png");
