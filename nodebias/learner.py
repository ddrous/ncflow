from ._utils import *


class Learner:
    def __init__(self, neuralnet, contexts, loss_fn_ctx, integrator, invariant=None, physics=None, key=None):

        # self.nb_envs = nb_envs
        # self.context_size = context_size

        self.nb_envs, self.context_size = contexts.params.shape

        self.neuralnet = neuralnet
        self.physics = physics
        self.invariant = invariant

        vectorfield = VectorField(neuralnet, physics)
        self.neuralode = NeuralODE(vectorfield, integrator, invariant)

        # ctx_key, loss_key = generate_new_keys(key, num=2)
        # self.contexts = ContextParams(self.nb_envs, self.context_size, key=ctx_key)
        self.contexts = contexts
        self.init_ctx_params = self.contexts.params.copy()

        self.loss_fn = lambda model, context, batch, weights: loss_fn(model, context, batch, weights, loss_fn_ctx, key=get_new_key(key))

    def save_learner(self, path):
        eqx.tree_serialise_leaves(path+"neuralode.eqx", self.neuralode)
        eqx.tree_serialise_leaves(path+"contexts.eqx", self.contexts)
        ## Save the loss function as well

    def load_learner(self, path):
        self.model = eqx.tree_deserialise_leaves(path+"neuralode.eqx", self.neuralode)
        self.context = eqx.tree_deserialise_leaves(path+"contexts.eqx", self.contexts)
















class ContextParams(eqx.Module):
    params: jnp.ndarray

    def __init__(self, nb_envs, context_size, key=None):
        if key is None:
            print("WARNING: No key provided for the context parameters. Initialising at 0.")
            self.params = jnp.zeros((nb_envs, context_size))

        else:
            self.params = jax.random.normal(get_new_key(key), (nb_envs, context_size))


class ID(eqx.Module):
    def __init__(self):
        pass
    def __call__(self, t, x):
        return jnp.zeros_like(x)

class VectorField(eqx.Module):
    physics: eqx.Module
    neuralnet: eqx.Module

    def __init__(self, neuralnet, physics=None):
        self.neuralnet = neuralnet
        self.physics = physics if physics is not None else ID()

    def __call__(self, t, x, ctx):
        # return self.physics(t, x) + self.neuralnet(t, x, ctx)
        return self.physics(t, x, ctx) + self.neuralnet(t, x, ctx)
        # return self.physics(t, x, ctx)


class NeuralODE(eqx.Module):
    vectorfield: VectorField
    integrator: callable
    # invariant: eqx.Module

    def __init__(self, vectorfield, integrator, invariant=None, key=None):
        self.vectorfield = vectorfield
        self.integrator = integrator
        # self.invariant = invariant

    def __call__(self, x0s, t_eval, ctx):

        # def integrate(x0):
        #     solution = diffrax.diffeqsolve(
        #                 diffrax.ODETerm(self.vectorfield),
        #                 diffrax.Tsit5(),
        #                 args=ctx,
        #                 t0=t_eval[0],
        #                 t1=t_eval[-1],
        #                 dt0=t_eval[1] - t_eval[0],
        #                 y0=x0,
        #                 stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-4),
        #                 saveat=diffrax.SaveAt(ts=t_eval),
        #                 max_steps=4096*1,
        #             )
        #     return solution.ys, solution.stats["num_steps"]

        # batched_ys, batched_num_steps = jax.vmap(integrate)(x0s)
        # return batched_ys, batched_num_steps

        rhs = lambda x, t: self.vectorfield(t, x, ctx)
        batched_ys = jax.vmap(rk4_integrator, in_axes=(None, 0, None))(rhs, x0s, t_eval)
        return batched_ys, t_eval.size


# def rk4_integrator(rhs, y0, t, rtol, atol, hmax, mxstep, max_steps_rev, kind):
def rk4_integrator(rhs, y0, t):
  def step(state, t):
    y_prev, t_prev = state
    h = t - t_prev
    k1 = h * rhs(y_prev, t_prev)
    k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
    k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
    k4 = h * rhs(y_prev + k3, t + h)
    y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y, t), y
  _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
  # return ys
  return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)
























def params_norm(params):
    """ norm of the parameters """
    return jnp.array([jnp.linalg.norm(x) for x in jax.tree_util.tree_leaves(params)]).sum()

def spectral_norm(params):
    """ spectral norm of the parameters """
    return jnp.array([jnp.linalg.svd(x, compute_uv=False)[0] for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]).sum()

def spectral_norm_estimation(model, nb_iters=5, *, key=None):
    """ estimating the spectral norm with the power iteration: https://arxiv.org/abs/1802.05957 """
    params = eqx.filter(model, eqx.is_array)
    matrices = [x for x in jax.tree_util.tree_leaves(params) if jnp.ndim(x)==2]
    nb_matrices = len(matrices)
    keys = generate_new_keys(key, num=nb_matrices)
    us = [jax.random.normal(k, (x.shape[0],)) for k, x in zip(keys, matrices)]
    vs = [jax.random.normal(k, (x.shape[1],)) for k, x in zip(keys, matrices)]

    for _ in range(nb_iters):
        for i in range(nb_matrices):
            vs[i] = matrices[i].T@us[i]
            vs[i] = vs[i] / jnp.linalg.norm(vs[i])
            us[i] = matrices[i]@vs[i]
            us[i] = us[i] / jnp.linalg.norm(us[i])

    sigmas = [u.T@x@v for x, u, v in zip(matrices, us, vs)]
    return jnp.array(sigmas).sum()

def infinity_norm_estimation(model, xs, ctx):
    xs_flat = jnp.reshape(xs, (-1, xs.shape[-1]))
    ys = jax.vmap(model, in_axes=(None, 0, None))(None, xs_flat, ctx)
    return jnp.mean(jnp.linalg.norm(ys, axis=-1) / jnp.linalg.norm(xs_flat, axis=-1))

def l2_norm_traj(xs, xs_hat):
    total_loss = jnp.mean((xs - xs_hat)**2, axis=-1)   ## TODO mean or sum ? Norm of d-dimensional vectors
    return jnp.sum(total_loss) / (xs.shape[-2] * xs.shape[-3])


def loss_fn(model, contexts, batch, weights, loss_fn_ctx, key=None):
    # print('\nCompiling function "loss_fn" ...\n')
    Xs, t_eval = batch
    print("Shapes of elements in a batch:", Xs.shape, t_eval.shape)

    all_loss, (all_nb_steps, all_term1, all_term2) = jax.vmap(loss_fn_ctx, in_axes=(None, 0, None, 0, None, None, None))(model, Xs[:, :, :, :], t_eval, contexts.params, 1e-0, 1e-3, key)

    total_loss = jnp.sum(all_loss*weights)

    return total_loss, (jnp.sum(all_nb_steps), all_term1, all_term2)
