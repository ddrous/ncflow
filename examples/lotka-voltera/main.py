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
nb_epochs = 100000



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

class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, key=None):
        key = generate_new_keys(key, num=1)[0]
        self.params = jax.random.uniform(key, (4,), minval=0.05, maxval=1.5)

    def __call__(self, t, x):
        dx0 = x[0]*self.params[0] - x[0]*x[1]*self.params[1]
        dx1 = x[0]*x[1]*self.params[3] - x[1]*self.params[2]
        return jnp.array([dx0, dx1])

# activation = jax.nn.softplus
activation = jax.nn.swish

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

        self.layers_context = [eqx.nn.Linear(context_size, width_size*2, key=keys[3]), activation,
                        eqx.nn.Linear(width_size*2, width_size*2, key=keys[11]), activation,
                        eqx.nn.Linear(width_size*2, width_size, key=keys[4]), activation,
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

        # ctx = jnp.zeros_like(y)     ## TODO: remove this line, please !

        # y = jnp.concatenate([y, context, y*context], axis=0)
        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)
        return y


physics = Physics(key=SEED)
augmentation = Augmentation(data_size=2, width_size=16*1, depth=3, context_size=context_size, key=SEED)
contexts = ContextParams(nb_envs, context_size, key=SEED)

# integrator = diffrax.Tsit5()
integrator = rk4_integrator

def loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, key):
    trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx)
    # term1 = l2_norm_traj(trajs, trajs_hat)
    # term1 = jnp.sum((trajs - trajs_hat)**2)     ## MSE TODO !
    # term1 = jnp.sum((trajs - trajs_hat)**2) / (trajs.shape[-2]*trajs.shape[-3])

    term1 = jnp.mean((trajs-trajs_hat)**2)

    term2_1 = spectral_norm_estimation(model.vectorfield.neuralnet, key=key)
    term2_2 = infinity_norm_estimation(model.vectorfield.neuralnet, trajs, ctx)
    term2 = term2_1 + alpha*term2_2

    loss_val = term1 + beta*term2
    return loss_val, (jnp.sum(nb_steps), term1, term2)


learner = Learner(augmentation, contexts, loss_fn_ctx, integrator, physics=physics, key=SEED)
# learner = Learner(augmentation, nb_envs, context_size, loss_fn_ctx, integrator, physics=None)



#%%

## Define optimiser and traine the model

# sched_node = optax.piecewise_constant_schedule(init_value=3e-2,
#                         boundaries_and_scales={int(nb_epochs*0.25):0.25, 
#                                                 int(nb_epochs*0.5):0.25,
#                                                 int(nb_epochs*0.75):0.25})
# sched_node = 1e-3
## exponential decay
sched_node = optax.exponential_decay(3e-4, nb_epochs*2, 0.99)
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
for propostion in np.linspace(0.25, 0.5, 2):
    trainer.dataloader.int_cutoff = int(propostion*nb_steps_per_traj)
    # nb_epochs = nb_epochs // 2 if nb_epochs > 1000 else 1000
    trainer.train(nb_epochs=nb_epochs, print_error_every=1000, update_context_every=1, save_path="tmp/", key=SEED)

#%%

## Test and visualise the results on a test dataloader

# raw_data = np.load("tmp/test_data.npz")
# dataset, t_eval = raw_data["X"][0:1, ...], raw_data["t"]
# test_dataloader = DataLoader(dataset, t_eval)

test_dataloader = DataLoader("tmp/test_data.npz")

visualtester = VisualTester(test_dataloader, trainer)

visualtester.visualise(int_cutoff=1.0, save_path="tmp/results.png");


#%%

## Print some other params
print("Parameters in the physics component:", learner.neuralode.vectorfield.physics.params)


#%%
# len(trainer.losses_node)