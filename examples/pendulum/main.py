from nodax import *



#%%

## Define dataloader for training

train_dataloader = DataLoader("tmp/simple_pendulum_big.npz", batch_size=32, int_cutoff=0.2, shuffle=True)

nb_envs = train_dataloader.nb_envs
nb_trajs_per_env = train_dataloader.nb_trajs_per_env
nb_steps_per_traj = train_dataloader.nb_steps_per_traj
data_size = train_dataloader.data_size

context_size = 2

#%%

## Define model and loss function for the learner


class Physics(eqx.Module):
    params: jnp.ndarray

    def __init__(self, key=None):
        keys = get_new_key(key, num=2)
        self.params = jnp.concatenate([jax.random.uniform(keys[0], (1,), minval=0.25, maxval=1.75),
                                       jax.random.uniform(keys[1], (1,), minval=1, maxval=30)])

    def __call__(self, t, x):
        L, g = self.params
        theta, theta_dot = x
        theta_ddot = -(g / L) * jnp.sin(theta)
        return jnp.array([theta_dot, theta_ddot])


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


physics = Physics()
augmentation = Augmentation(data_size=2, width_size=32, depth=3, context_size=2)

integrator = diffrax.Tsit5()


def loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, key):
    trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx)
    term1 = l2_norm_traj(trajs, trajs_hat)

    term2_1 = spectral_norm_estimation(model.vectorfield.augmentation, key=key)
    term2_2 = infinity_norm_estimation(model.vectorfield.augmentation, trajs, ctx)
    term2 = term2_1 + alpha*term2_2

    loss_val = term1 + beta*term2
    return loss_val, (jnp.sum(nb_steps), term1, term2)


learner = Learner(augmentation, nb_envs, context_size, loss_fn_ctx, integrator, physics=physics)



#%%

## Define optimiser and traine the model

sched_node = optax.piecewise_constant_schedule(init_value=3e-3,
                        boundaries_and_scales={int(2000*0.25):0.25, 
                                                int(2000*0.5):0.25,
                                                int(2000*0.75):0.25})
sched_ctx = optax.piecewise_constant_schedule(init_value=3e-3,
                        boundaries_and_scales={int(2000*0.25):0.25, 
                                                int(2000*0.5):0.25,
                                                int(2000*0.75):0.25})

opt_node = optax.adabelief(sched_node)
opt_ctx = optax.adabelief(sched_ctx)

trainer = Trainer(train_dataloader, learner, (opt_node, opt_ctx))

trainer.train(nb_epochs=2, update_context_every=1, save_path="tmp/", key=jax.random.PRNGKey(0))

#%%

## Test and visualise the results on a test dataloader

test_dataloader = DataLoader("tmp/simple_pendulum_small.npz", batch_size=1, int_cutoff=0.2, shuffle=False)
visualtester = VisualTester(test_dataloader, trainer)

visualtester.visualise(cutoff=False, save_path="tmp/alternating_node.png")


#%%
