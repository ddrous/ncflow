
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import os

import diffrax
from ncf import RK4
import jax.numpy as jnp

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the (unused) seed
import argparse

if _in_ipython_session:
	args = argparse.Namespace(split='test', savepath="data/", verbose=1)
else:
	parser = argparse.ArgumentParser(description='Data Generation for Lotka-Volterra System')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Save location of the created data', default='data/', required=False)
	parser.add_argument('--verbose', type=int, help='Whether to print details or not ?', default=1, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"
savepath = args.savepath

if args.verbose != 0:
  print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)
  print('=== Parsed arguments to generate data ===')
  print(' Split:', split)
  print(' Savepath:', savepath)
  print()

#%%


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import jax.numpy as jnp

# Define the Lotka-Volterra system
def lotka_volterra(t, state, alpha, beta, delta, gamma):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]
    # return jnp.array([dx_dt, dy_dt])

if split == "train" or split=="test":
  # Training environments
  environments = [
      {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
      {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
      {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
      {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
      {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
      {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
      {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
      {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
      {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},
  ]
elif split == "adapt" or split == "adapt_test":
  ## Adaptation environments
  environments = [
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 1.125},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 1.125},
  ]
elif split == "adapt_huge":
  environments = [
      {"alpha": 0.5, "beta": b, "gamma": 0.5, "delta": d} for b in np.linspace(0.25, 1.25, 1) for d in np.linspace(0.25, 1.25, 1)]

if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test" or split == "adapt_test":
  n_traj_per_env = 32     ## testing
elif split == "adapt" or split == "adapt_huge":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(10/0.5)

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)

max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    # Initial conditions (prey and predator concentrations)
    np.random.seed(j if not split in ["test", "adapt_test"] else max_seed - j)
    initial_state = np.random.uniform(size=(2,)) + 1.

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(lotka_volterra, t_span, initial_state, args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

        # Solve the ODEs using a custom RK4 integrator
        # ys = RK4(lotka_volterra, 
        #             (t_eval[0], t_eval[-1]),
        #             initial_state,
        #            *(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), 
        #             t_eval=t_eval, 
        #             subdivisions=5)
        # data[i, j, :, :] = ys

## Check if savepath exists, if not create it
if not os.path.exists(savepath):
    os.makedirs(savepath)

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train.npz'
elif split == "test":
  filename = savepath+'test.npz'
elif split == "adapt":
  filename = savepath+'ood_train.npz'
elif split == "adapt_test":
  filename = savepath+'ood_test.npz'
elif split == "adapt_huge":
  filename = savepath+'grid_train.npz'

np.savez(filename, t=t_eval, X=data)












# %%
if _in_ipython_session:
  ### Extract the solution and create an animation of the Lotka-Volterra system ###

  prey_concentration, predator_concentration = solution.y
  # prey_concentration, predator_concentration = ys[...,0], ys[...,1]

  fig, ax = plt.subplots()
  eps = 0.5
  ax.set_xlim(-eps, np.max(prey_concentration)+eps)
  ax.set_ylim(-eps, np.max(predator_concentration)+eps)
  ax.set_xlabel('Preys')
  ax.set_ylabel('Predators')

  concentrations, = ax.plot([], [], 'r-', lw=1, label='Concentrations')
  time_template = 'Time = %.1fs'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  ax.legend()

  def animate(i):
      concentrations.set_data(prey_concentration[:i], predator_concentration[:i])
      time_text.set_text(time_template % t_eval[i])
      return concentrations, time_text

  ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=5, repeat=False, blit=True)
  plt.show()

  ## Save the movie to a small gif file
  ani.save('./data/lotka_volterra.gif', fps=30)
