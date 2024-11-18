from ._utils import *


class Learner:
    def __init__(self, vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=None):

        # self.nb_envs = nb_envs
        # self.context_size = context_size

        self.nb_envs, self.context_size = contexts.params.shape

        # self.neuralnet = neuralnet
        # self.physics = physics
        # self.invariant = invariant

        # vectorfield = VectorField(neuralnet, physics)
        # self.neuralode = NeuralODE(vectorfield, integrator, invariant)
        self.neuralode = NeuralODE(vectorfield, integrator, ivp_args)      ## TODO call this Universal ODE

        # ctx_key, loss_key = generate_new_keys(key, num=2)
        # self.contexts = ContextParams(self.nb_envs, self.context_size, key=ctx_key)
        self.contexts = contexts
        self.init_ctx_params = self.contexts.params.copy()

        # self.loss_fn = lambda model, contexts, batch, weights: loss_fn(model, contexts, batch, weights, loss_fn_ctx, key=get_new_key(key))
        self.loss_fn = lambda model, contexts, batch, weights, key: loss_fn(model, contexts, batch, weights, loss_fn_ctx, key)

    def save_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"

        eqx.tree_serialise_leaves(path+"neuralode.eqx", self.neuralode)
        eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)

        np.save(path+"contexts_init.npy", self.init_ctx_params)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"

        self.neuralode = eqx.tree_deserialise_leaves(path+"neuralode.eqx", self.neuralode)
        self.contexts = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)

        self.init_ctx_params = np.load(path+"contexts_init.npy")













class ContextParams(eqx.Module):
    params: jnp.ndarray

    def __init__(self, nb_envs, context_size, key=None):
        if key is None:
            print("WARNING: No key provided for the context initialization. Initializing at 0.")
            self.params = jnp.zeros((nb_envs, context_size))

        else:
            self.params = jax.random.normal(get_new_key(key), (nb_envs, context_size))


class NoPhysics(eqx.Module):
    def __init__(self):
        pass
    def __call__(self, t, x, *args):
        return jnp.zeros_like(x)



class DefaultVectorField(eqx.Module):
    physics: eqx.Module
    neuralnet: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics if physics is not None else NoPhysics()

    def __call__(self, t, x, ctx, ctx_):
        return self.physics(t, x, ctx) + self.augmentation(t, x, ctx)










# class NeuralContextFlow(eqx.Module):
class NeuralODE(eqx.Module):
    vectorfield: eqx.Module
    integrator: callable
    ivp_args: dict

    def __init__(self, vectorfield, integrator, ivp_args, key=None):
        self.integrator = integrator
        self.ivp_args = ivp_args
        self.vectorfield = vectorfield
        # # self.vectorfield = lambda t, x, ctxs: vectorfield(t, x, *ctxs)
        # self.vectorfield = lambda t, x, ctxs: vectorfield(t, x, ctxs[0], ctxs[1])

    def __call__(self, x0s, t_eval, ctx, ctx_):

        if isinstance(self.integrator, type(eqx.Module)):
                def integrate(y0): 
                    sol = diffrax.diffeqsolve(
                            diffrax.ODETerm(self.vectorfield),
                            self.integrator(),
                            args=(ctx, ctx_.squeeze()),
                            t0=t_eval[0],
                            t1=t_eval[-1],
                            dt0=self.ivp_args["dt_init"],
                            y0=y0,
                            stepsize_controller=diffrax.PIDController(rtol=self.ivp_args.get("rtol", 1e-3), 
                                                                      atol=self.ivp_args.get("atol", 1e-6)),
                            saveat=diffrax.SaveAt(ts=t_eval),
                            adjoint=self.ivp_args.get("adjoint", diffrax.RecursiveCheckpointAdjoint()),
                            max_steps=self.ivp_args.get("max_steps", 4096*1)
                        )
                    return sol.ys, sol.stats["num_steps"]

        else:   ## Custom-made integrator
            def integrate(y0):
                ys = self.integrator(self.vectorfield, 
                                     (t_eval[0], t_eval[-1]), 
                                     y0,
                                     (ctx, ctx_.squeeze()), 
                                     t_eval=t_eval, 
                                     **self.ivp_args)
                return ys, t_eval.size

        return jax.vmap(integrate)(x0s)








def RK4(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """ Perform numerical integration with a time step divided by the evaluation subdivision factor (Not necessarily equally spaced). If we get NaNs, we can try to increasing the subdivision factor for finer time steps."""
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        k1 = h * fun(t_prev, y_prev, *args)
        k2 = h * fun(t_prev + h/2., y_prev + k1/2., *args)
        k3 = h * fun(t_prev + h/2., y_prev + k2/2., *args)
        k4 = h * fun(t + h, y_prev + k3, *args)
        y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]

def Euler(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """ Euler integrator"""
    if t_eval is None:
        if t_span[0] is None:
            t_eval = jnp.array([t_span[1]])
            raise Warning("t_span[0] is None. Setting t_span[0] to 0.")
        elif t_span[1] is None:
            raise ValueError("t_span[1] must be provided if t_eval is not.")
        else:
            t_eval = jnp.array(t_span)

    hs = t_eval[1:] - t_eval[:-1]
    t_ = t_eval[:-1, None] + jnp.arange(subdivisions)[None, :]*hs[:, None]/subdivisions
    t_solve = jnp.concatenate([t_.flatten(), t_eval[-1:]])
    eval_indices = jnp.arange(0, t_solve.size, subdivisions)

    def step(state, t):
        t_prev, y_prev = state
        h = t - t_prev
        y = y_prev + h * fun(t_prev, y_prev, *args)
        return (t, y), y

    _, ys = jax.lax.scan(step, (t_solve[0], y0), t_solve[:])
    return ys[eval_indices, :]





# @partial(jax.jit, static_argnames=("fun", "t_span", "t_eval", "dt_max"))
# @partial(eqx.filter_jit, static_argnames=("fun", "t_span", "t_eval", "dt_max"))
# def RK4(fun, t_span, y0, *args, t_eval=None, dt_max=1e-2, **kwargs):
#     t_eval = jnp.array(t_span[1]) if t_eval is None else t_eval
#     t_solve = jnp.arange(t_span[0], t_span[1], dt_max)

#     t_all = jnp.insert(t_solve, jnp.searchsorted(t_solve, t_eval), t_eval)
#     eval_indices = jnp.searchsorted(t_all, t_eval)
  
#     def step(t, state):
#         y_prev, t_prev = state
#         h = t - t_prev
#         k1 = h * fun(t_prev, y_prev, *args)
#         k2 = h * fun(t_prev + h/2., y_prev + k1/2., *args)
#         k3 = h * fun(t_prev + h/2., y_prev + k2/2., *args)
#         k4 = h * fun(t + h, y_prev + k3, *args)
#         y = y_prev + (k1 + 2*k2 + 2*k3 + k4) / 6.
#         return (y, t), y

#     _, ys = jax.lax.scan(step, (t_all[0], y0), t_all[:])
#     return ys[eval_indices, :]


















# def loss_fn(model, contexts, batch, weights, loss_fn_ctx, key=None):
#     # print('\nCompiling function "loss_fn" ...\n')
#     Xs, t_eval = batch
#     print("Shapes of elements in a batch:", Xs.shape, t_eval.shape)

#     all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_fn_ctx, in_axes=(None, 0, None, 0, None, None, None))(model, Xs[:, :, :, :], t_eval, contexts.params, 1e-0, 1e-3, key)

#     total_loss = jnp.sum(all_loss*weights)

#     return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)



# def loss_fn_cf(model, contexts, batch, weights, loss_fn_ctx, key=None):
#     # print('\nCompiling function "loss_fn" ...\n')
#     Xs, t_eval = batch
#     print("Shapes of elements in a batch:", Xs.shape, t_eval.shape)

#     all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_fn_ctx, in_axes=(None, 0, None, 0, None, None, None, None))(model, Xs[:, :, :, :], t_eval, contexts.params, 1e-0, 1e-3, contexts.params, key)

#     total_loss = jnp.sum(all_loss*weights)
#     # total_loss = jnp.sum(all_loss)

#     return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)



def loss_fn(model, contexts, batch, weights, loss_fn_ctx, key):
    # print('\nCompiling function "loss_fn" ...\n')
    Xs, t_eval = batch
    print("Shapes of elements in a batch:", Xs.shape, t_eval.shape)

    all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_fn_ctx, in_axes=(None, 0, None, 0, None, None))(model, Xs[:, :, :, :], t_eval, contexts.params, contexts.params, key)

    recons = jnp.sum(all_loss*weights)
    # recons = jnp.sum(all_loss)
    # recons = jnp.max(all_loss)  # simply return the max, then things should even out naturally

    # regul = 1e-5*params_norm(eqx.filter(model, eqx.is_array))
    # regul = 1e-5*spectral_norm_estimation(model)
    regul = 0.

    total_loss = recons + regul

    return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)







def context_flow_loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, ctx_, key):
    trajs_hat, nb_steps = jax.vmap(model, in_axes=(None, None, None, 0))(trajs[:, 0, :], t_eval, ctx, ctx_)
    new_trajs = jnp.broadcast_to(trajs, trajs_hat.shape)

    term1 = jnp.mean((new_trajs-trajs_hat)**2)

    term2 = alpha*jnp.mean((ctx)**2)

    loss_val = term1 + beta*term2
    return loss_val, (jnp.sum(nb_steps)/ctx_.shape[0], term1, term2)



def default_loss_fn_ctx(model, trajs, t_eval, ctx, alpha, beta, ctx_, key):
    trajs_hat, nb_steps = model(trajs[:, 0, :], t_eval, ctx, ctx)

    term1 = jnp.mean((trajs-trajs_hat)**2)

    term2_1 = spectral_norm_estimation(model.vectorfield.augmentation, key=key)
    term2_2 = infinity_norm_estimation(model.vectorfield.augmentation, trajs, ctx)
    term2 = term2_1 + alpha*term2_2

    loss_val = term1 + beta*term2
    return loss_val, (jnp.sum(nb_steps), term1, term2)
