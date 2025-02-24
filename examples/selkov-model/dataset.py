
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import diffrax
from ncf import RK4
import jax.numpy as jnp

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	# args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	args = argparse.Namespace(split='adapt_test', savepath="tmp/", seed=2026, verbose=1)
else:
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='tmp/', required=False)
	parser.add_argument('--seed',type=int, help='Seed to gnerate the data', default=2026, required=False)
	parser.add_argument('--verbose',type=int, help='Whether to print details or not ?', default=1, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"

savepath = args.savepath
seed = args.seed

if args.verbose != 0:
  print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)
  print('=== Parsed arguments to generate data ===')
  print(' Split:', split)
  print(' Savepath:', savepath)
  print(' Seed:', seed)
  print()


## Set numpy seed for reproducibility
np.random.seed(seed)


#%%

# Image(filename="tmp/coda_dataset.png")


#%%


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# import jax
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
# import diffrax

# Define the Selkov system: https://en.wikipedia.org/wiki/Hopf_bifurcation
def selkov(t, y, a, b):
    x, y = y
    dx = -x + a*y + (x**2)*y
    dy = b - a*y - (x**2)*y
    return np.array([dx, dy])

# def selkov(t, y, a, b):
#     x, y = y
#     dx = 0.5*x - 0.5*x*y
#     dy = 0.5*x*y - 0.5*y
#     return np.array([dx, dy])


""" Earlier implementation of RK4 integrator. See better implementation in nodax.RK4"""
# def rk4_integrator(rhs, y0, t):
#   def step(state, t):
#     y_prev, t_prev = state
#     h = t - t_prev
#     k1 = h * rhs(y_prev, t_prev)
#     k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
#     k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
#     k4 = h * rhs(y_prev + k3, t + h)
#     y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
#     return (y, t), y

#   y = y0
#   ys = [y]
#   for i in range(t.size-1):
#     _, y = step((y, t[i]), t[i+1])
#     ys.append(y)
#   return  jnp.vstack(ys)

#   _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
#   return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)



if split == "train" or split=="test":
  # Training environments
  # environments = [(0.1, b) for b in np.linspace(-1.0, 1.0, 21)]

  environments = [(0.1, b) for b in list(np.linspace(-1, -0.25, 7))\
        + list(np.linspace(-0.1, 0.1, 7))\
        + list(np.linspace(0.25, 1., 7))]

  # environments = [(0.1, b) for b in np.linspace(0.2, 1.0, 16)]
  # environments = [(0.1, b) for b in np.linspace(0.5, 0.6, 1)]
elif split == "adapt" or split == "adapt_test":
  ## Adaptation environments
  # environments = [(0.1, b) for b in np.linspace(-1.25, 1.25, 21)[::4]]
  # environments = [(0.1, b) for b in np.linspace(0.1, 1.1, 16)[::4]]
  environments = [(0.1, b) for b in [-1.25, -0.65, -0.05, 0.02, 0.6, 1.2]]

# elif split == "adapt_huge":
#   environments = [
#       {"alpha": 0.5, "beta": b, "gamma": 0.5, "delta": d} for b in np.linspace(0.25, 1.25, 1) for d in np.linspace(0.25, 1.25, 1)]

# ## Lots of data environment
# environments = []
# for beta in np.linspace(0.5, 1.5, 11):
#   new_env = {"alpha": 0.5, "beta": beta, "gamma": 0.5, "delta": 0.5}
#   environments.append(new_env)



if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 4     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = 11

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 40)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split not in ["test", "adapt_test", "adapt_huge"] else max_seed - j)
    initial_state = np.random.uniform(0, 3, 2)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        # initial_state = [0, 2]

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(selkov, t_span, initial_state, args=(selected_params[0], selected_params[1]), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

        # rhs = lambda x, t: lotka_volterra(t, x, selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"])
        # solution = rk4_integrator(rhs, initial_state, t_eval)
        # data[i, j, :, :] = solution

        # ys = RK4(selkov, 
        #             (t_eval[0], t_eval[-1]),
        #             initial_state,
        #            *(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), 
        #             t_eval=t_eval, 
        #             subdivisions=5)
        # data[i, j, :, :] = ys

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'
elif split == "adapt_test":
  filename = savepath+'adapt_test_data.npz'
elif split == "adapt_huge":
  filename = savepath+'adapt_huge_data.npz'

np.savez(filename, t=t_eval, X=data)












# %%
if _in_ipython_session:
  a = .1
  y0 = [0, 2]
  # y0 = [1, 1]
  t = np.linspace(*t_span, 1000)

  for b in np.linspace(0., 1.2, 15)[:]:
      t_span = [0, 1000]
      sol = solve_ivp(selkov, t_span, y0, args=(a, b), dense_output=True)
      y = sol.sol(t)
      plt.plot(y[0], y[1], label=f'b={b:.2f}')

  plt.xlabel('x')
  plt.ylabel('y')
  plt.legend()
  plt.show()
