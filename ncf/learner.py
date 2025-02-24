from ._utils import *




class Learner:
    """
    A learner to contain both the shared params (the neural_ode),
    the the environment-specific params (the contexts) and the loss function,
    the loss function to mer, the contexts, the batch, the weights and the key.
    """

    def __init__(self, vectorfield, contexts, loss_fn_ctx, integrator, ivp_args, key=None):
        self.nb_envs, self.context_size = contexts.params.shape
        self.neuralode = NeuralODE(vectorfield, integrator, ivp_args)

        self.contexts = contexts
        self.init_ctx_params = self.contexts.params.copy()

        self.loss_fn = lambda model, contexts, batch, weights, key: loss_fn(model, contexts, batch, weights, loss_fn_ctx, key)

    def save_learner(self, path, suffix=None):
        assert path[-1] == "/", "ERROR: Invalid id parovided. The path must end with /"

        if suffix is None:
            eqx.tree_serialise_leaves(path+"neuralode.eqx", self.neuralode)
            eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)
            np.save(path+"contexts_init.npy", self.init_ctx_params)

        else:
            eqx.tree_serialise_leaves(f"{path}/neuralode_{suffix}.eqx", self.neuralode)
            eqx.tree_serialise_leaves(f"{path}/contexts_{suffix}.eqx", self.contexts)

    def load_learner(self, path):
        assert path[-1] == "/", "ERROR: Invalidn parovided. The path must end with /"

        self.neuralode = eqx.tree_deserialise_leaves(path+"neuralode.eqx", self.neuralode)
        self.contexts = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)

        self.init_ctx_params = np.load(path+"contexts_init.npy")

    def turn_off_self_modulation(self):
        """ Turn off the self-modulation of the vector field """
        new_model = eqx.tree_at(lambda m: m.vectorfield.taylor_order, self.neuralode, 0)    ## Copy of the model
        return new_model







#### MODULES ####

class Swish(eqx.Module):
    """Swish activation function"""
    beta: jnp.ndarray
    def __init__(self, key=None):
        self.beta = jax.random.uniform(key, shape=(1,), minval=0.01, maxval=1.0)
    def __call__(self, x):
        return x * jax.nn.sigmoid(self.beta * x)


class SelfModulatedVectorField(eqx.Module):
    """ 
    Contextual Self-Modulation: Taylor expansion of the vector field (vf) about the context vector.
    The taylor_order is the order k of the expansion : https://openreview.net/forum?id=8vzMLo8LDN
    The vector field is the sum of the physics (if available) and the augmentation (or the neuralnet) : https://arxiv.org/abs/2010.04456
    """
    physics: eqx.Module
    augmentation: eqx.Module
    taylor_order: int

    def __init__(self, augmentation, physics=None, taylor_order=0):
        self.augmentation = augmentation
        self.physics = physics
        self.taylor_order = taylor_order

    def __call__(self, t, x, ctxs):
        ctx, ctx_ = ctxs

        if self.physics is None:
            vf = lambda xi: self.augmentation(t, x, xi)
        else:
            vf = lambda xi: self.physics(t, x, xi) + self.augmentation(t, x, xi)

        if self.taylor_order == 0:
            return vf(ctx_)
        else:
            gradvf = lambda xi_: eqx.filter_jvp(vf, (xi_,), (ctx-xi_,))[1]
            if self.taylor_order == 1:
                return vf(ctx_) + 1.0*gradvf(ctx_)
            elif self.taylor_order == 2:
                scd_order_term = eqx.filter_jvp(gradvf, (ctx_,), (ctx-ctx_,))[1]
                return vf(ctx_) + 1.5*gradvf(ctx_) + 0.5*scd_order_term
            else:
                raise ValueError(f"Taylor order must be 0, 1 or 2. Got: {self.taylor_order}. For higher orders, see https://docs.jax.dev/en/latest/jax.experimental.jet.html")


class ContextParams(eqx.Module):
    """ The context vectors for all environments in one tensor """
    params: jnp.ndarray

    def __init__(self, nb_envs, context_size, key=None):
        if key is None:
            print("WARNING: No key provided for the context initialization. Initializing at 0.")
            self.params = jnp.zeros((nb_envs, context_size))

        else:
            self.params = jax.random.normal(get_new_key(key), (nb_envs, context_size))


class NoPhysics(eqx.Module):
    """ A dummy physics module that returns zeros """
    def __init__(self):
        pass
    def __call__(self, t, x, *args):
        return jnp.zeros_like(x)


class DefaultVectorField(eqx.Module):
    """ Example of a vector field as the sum of the physics and the neuralnet """
    physics: eqx.Module
    neuralnet: eqx.Module

    def __init__(self, augmentation, physics=None):
        self.augmentation = augmentation
        self.physics = physics if physics is not None else NoPhysics()

    def __call__(self, t, x, ctx, ctx_):
        return self.physics(t, x, ctx) + self.augmentation(t, x, ctx)


class NeuralODE(eqx.Module):
    """ A neural ODE module that integrates the differential equation to generate candidate trajectories """
    vectorfield: eqx.Module
    integrator: callable
    ivp_args: dict

    def __init__(self, vectorfield, integrator, ivp_args):
        self.integrator = integrator
        self.ivp_args = ivp_args
        self.vectorfield = vectorfield

    def __call__(self, x0s, t_eval, ctx, ctx_):

        if isinstance(self.integrator, type(eqx.Module)):       ## If integrator is from diffrax
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

        else:                                                   ## If integrator is a custom function
            def integrate(y0):
                ys = self.integrator(self.vectorfield, 
                                     (t_eval[0], t_eval[-1]), 
                                     y0,
                                     (ctx, ctx_.squeeze()), 
                                     t_eval=t_eval, 
                                     **self.ivp_args)
                return ys, t_eval.size

        return jax.vmap(integrate)(x0s)







#### INTEGRATORS / SOLVERS ####

def RK4(fun, t_span, y0, *args, t_eval=None, subdivisions=1, **kwargs):
    """ Custom RK4 integrator: performs numerical integration with a fixed time step equal to the evaluation time step 
        divided by the 'subdivision' argument (Not necessarily equally spaced). If we get NaNs, we can try to increasing 
        the subdivision argument for finer time steps."""
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
    """ Custom Euler integrator with a subdivision factor as in RK4 """
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







#### LOSS FUNCTIONS ####

def loss_fn(model, contexts, batch, weights, loss_fn_ctx, key):
    """ Aggregrate loss function for all environments in the dataset
        Each environment has its own context vector, and its loss is weighted (typically uniform coefficients)
    """

    # print('\nCompiling function "loss_fn" ...\n')
    Xs, t_eval = batch
    print("Shapes of elements in a batch:", Xs.shape, t_eval.shape, "\n")

    all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_fn_ctx, in_axes=(None, 0, None, 0, None, None))(model, 
                                                                                                                   Xs[:, :, :, :], 
                                                                                                                   t_eval, 
                                                                                                                   contexts.params, 
                                                                                                                   contexts.params, key)

    total_loss = jnp.sum(all_loss*weights)

    return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)
