
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image


try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	# args = argparse.Namespace(split='test', savepath="./runs/24012024-084802/", seed=3422)
else:
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--split', type=str, help='Generate "train", "test" or "adapt" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='tmp/', required=False)
	parser.add_argument('--seed',type=int, help='Seed to gnerate the data', default=42, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt"], "Split must be either 'train', 'test' or 'adapt'"

savepath = args.savepath
seed = args.seed

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

# Define the Lotka-Volterra system
keys = ['J0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'K1', 'q', 'N', 'A', 'kappa', 'psi', 'k']

def lotka_volterra(t, x, params):
    J0, k1, k2, k3, k4, k5, k6, K1, q, N, A, kappa, psi, k = [params[k] for k in keys]

    d = np.zeros(7)
    k1s1s6 = k1 * x[0] * x[5] / (1 + (x[5]/K1) ** q)
    d[0] = J0 - k1s1s6
    d[1] = 2 * k1s1s6 - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
    d[2] = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
    d[3] = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
    d[4] = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4] 
    d[5] = -2 * k1s1s6 + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
    d[6] = psi * kappa * (x[3] - x[6]) - k * x[6]

    return d



def rk4_integrator(rhs, y0, t):
  def step(state, t):
    y_prev, t_prev = state
    h = t - t_prev
    k1 = h * rhs(y_prev, t_prev)
    k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
    k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
    k4 = h * rhs(y_prev + k3, t + h)
    y = y_prev + 1./6. * (k1 + 2 * k2 + 2 * k3 + k4)
    return (y, t), y

  y = y0
  ys = [y]
  for i in range(t.size-1):
    _, y = step((y, t[i]), t[i+1])
    ys.append(y)
  return  jnp.vstack(ys)

#   _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
#   return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)



if split == "train" or split=="test":
  # Training environments

    k1_range = [100, 90, 80]  
    K1_range = [1, 0.75, 0.5]
    environments = [{'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]


elif split == "adapt":
  ## Adaptation environments
    k1_range = [85, 95]  
    K1_range = [0.625, 0.875]
    environments = [{'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]




if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(1.0/0.05)
# n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 7))

# Time span for simulation
t_span = (0, 1)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

ic_range = [(0.15, 1.60), (0.19, 2.16), (0.04, 0.20), (0.10, 0.35), (0.08, 0.30), (0.14, 2.67), (0.05, 0.10)]

for j in range(n_traj_per_env):

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        # np.random.seed(index if not self.test else self.max - index)
        initial_state = np.random.random(7) * np.array([b-a for a, b in ic_range]) + np.array([a for a, _ in ic_range])

        # print("Initial state", initial_state)

        # Solve the ODEs using SciPy's solve_ivp
        # solution = solve_ivp(lotka_volterra, t_span, initial_state, args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), t_eval=t_eval)
        # data[i, j, :, :] = solution.y.T

        rhs = lambda x, t: lotka_volterra(t, x, selected_params)
        solution = rk4_integrator(rhs, initial_state, t_eval)
        data[i, j, :, :] = solution

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'

np.savez(filename, t=t_eval, X=data)













if _in_ipython_session:
  # Extract and plot the gycolytic oscillator
  solution = data[0, 0, :, :]
  t = t_eval
  plt.plot(t, solution[:, 0], label="S1")
  plt.plot(t, solution[:, 1], label="S2")
  plt.plot(t, solution[:, 2], label="S3")
  plt.plot(t, solution[:, 3], label="S4")
  plt.plot(t, solution[:, 4], label="S5")
  plt.plot(t, solution[:, 5], label="S6")
  plt.plot(t, solution[:, 6], label="S7")
  plt.xlabel("Time")
  plt.ylabel("Concentration")
  plt.legend()
  plt.show()
