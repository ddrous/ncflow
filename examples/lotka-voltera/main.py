from nodebias import *
from IPython.display import Image

# jax.config.update("jax_debug_nans", True)


#%%

Image(filename="tmp/coda_lv_setting_1.png")
#%%
Image(filename="tmp/coda_lv_setting_2.png")

## Additionally
# LV, GO: 4-layer MLPs with hidden layers of width 64
# We apply Swish activation (Ramachandran et al., 2018).
# The hypernet A is a single affine layer NN

#%%


## Hyperparams
SEED = 3
context_size = 2
nb_epochs = 2000



## Define dataloader for training

# train_dataloader = DataLoader("tmp/dataset_big.npz", batch_size=-1, int_cutoff=0.2, shuffle=True)
train_dataloader = DataLoader("tmp/train_data.npz", batch_size=-1, int_cutoff=0.5, shuffle=True)

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


class Augmentation(eqx.Module):
    layers_data: list
    layers_context: list
    layers_shared: list

    def __init__(self, data_size, width_size, depth, context_size, key=None):
        keys = generate_new_keys(key, num=12)
        self.layers_data = [eqx.nn.Linear(data_size, width_size, key=keys[0]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[10]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[1]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[2])]

        self.layers_context = [eqx.nn.Linear(context_size, width_size*2, key=keys[3]), jax.nn.softplus,
                        eqx.nn.Linear(width_size*2, width_size*2, key=keys[11]), jax.nn.softplus,
                        eqx.nn.Linear(width_size*2, width_size, key=keys[4]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[5])]

        self.layers_shared = [eqx.nn.Linear(data_size*2, width_size, key=keys[6]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[7]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, width_size, key=keys[8]), jax.nn.softplus,
                        eqx.nn.Linear(width_size, data_size, key=keys[9])]


    def __call__(self, t, x, ctx):

        y = x
        ctx = ctx
        for i in range(len(self.layers_data)):
            y = self.layers_data[i](y)
            ctx = self.layers_context[i](ctx)

        # y = jnp.concatenate([y, context, y*context], axis=0)
        y = jnp.concatenate([y, ctx], axis=0)
        for layer in self.layers_shared:
            y = layer(y)
        return y


physics = Physics(key=SEED)
augmentation = Augmentation(data_size=2, width_size=32, depth=3, context_size=2, key=SEED)
contexts = ContextParams(nb_envs, context_size, key=SEED)

integrator = diffrax.Tsit5()

def loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, key):
    trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx)
    # term1 = l2_norm_traj(trajs, trajs_hat)
    term1 = jnp.sum((trajs - trajs_hat)**2)     ## MSE TODO !

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
sched_node = 3e-3
# sched_ctx = optax.piecewise_constant_schedule(init_value=3e-2,
#                         boundaries_and_scales={int(nb_epochs*0.25):0.25, 
#                                                 int(nb_epochs*0.5):0.25,
#                                                 int(nb_epochs*0.75):0.25})
sched_ctx = 3e-3

opt_node = optax.adabelief(sched_node)
opt_ctx = optax.adabelief(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx), key=SEED)

trainer.train(nb_epochs=nb_epochs, update_context_every=1, save_path="tmp/", key=SEED)

#%%

## Test and visualise the results on a test dataloader

test_dataloader = DataLoader("tmp/test_data.npz", batch_size=1, shuffle=False)
visualtester = VisualTester(test_dataloader, trainer)

visualtester.visualise(int_cutoff=1.0, save_path="tmp/results.png");


#%%

## Print some other params
print("Parameters in the physics component:", learner.neuralode.vectorfield.physics.params)
