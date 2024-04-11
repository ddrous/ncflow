
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import diffrax
from nodax import RK4
# import jax.numpy as jnp
import numpy as jnp   ## Ugly, just cuz I don't wanna change the code below

## Set jax platform to CPU
import jax
# jax.config.update('jax_platform_name', 'cpu')

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	# args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	args = argparse.Namespace(split='test', savepath="./tmp/", seed=2026, verbose=1)
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

# Define the Lotka-Volterra system
keys = ['J0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'K1', 'q', 'N', 'A', 'kappa', 'psi', 'k']

def glycolytic_oscilator(t, x, params):
    J0, k1, k2, k3, k4, k5, k6, K1, q, N, A, kappa, psi, k = [params[0][k] for k in keys]

    # d = np.zeros(7)
    k1s1s6 = k1 * x[0] * x[5] / (1 + (x[5]/K1) ** q)
    # print("k1s1s6", k1s1s6, "JO", J0, "\n\n")
    d0 = J0 - k1s1s6
    d1 = 2 * k1s1s6 - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
    d2 = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
    d3 = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
    d4 = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4] 
    d5 = -2 * k1s1s6 + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
    d6 = psi * kappa * (x[3] - x[6]) - k * x[6]

    return jnp.array([d0, d1, d2, d3, d4, d5, d6])


#   _, ys = jax.lax.scan(step, (y0, t[0]), t[1:])
#   return jnp.concatenate([y0[jnp.newaxis, :], ys], axis=0)



if split == "train" or split=="test":
  # Training environments
    k1_range = [100, 90, 80]
    K1_range = [1, 0.75, 0.5]
    environments = [{'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]


elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  ## Adaptation environments
    k1_range = [85, 95]
    K1_range = [0.625, 0.875]
    # k1_range = [85]
    # K1_range = [0.625]
    environments = [{'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]




if split == "train":
  n_traj_per_env = 32     ## training
elif split == "test" or split == "adapt_test":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(1.0/0.05)
# n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 7))
# data = jnp.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 7))

# Time span for simulation
t_span = (0, 1)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames

ic_range = [(0.15, 1.60), (0.19, 2.16), (0.04, 0.20), (0.10, 0.35), (0.08, 0.30), (0.14, 2.67), (0.05, 0.10)]
max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split =="test" else max_seed - j)
    initial_state = np.random.random(7) * np.array([b-a for a, b in ic_range]) + np.array([a for a, _ in ic_range])

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        # print("Initial state", initial_state)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(glycolytic_oscilator, t_span, initial_state, args=((selected_params,),), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

        # # use diffrax instead, with the DoPri5 integrator
        # solution = diffrax.diffeqsolve(diffrax.ODETerm(glycolytic_oscilator),
        #                                diffrax.Dopri5(),
        #                                args=(selected_params,),
        #                                t0=t_span[0],
        #                                t1=t_span[1],
        #                               #  dt0=t_eval[1]-t_eval[0],
        #                                dt0=1e-4,
        #                                y0=initial_state,
        #                                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-7),
        #                                saveat=diffrax.SaveAt(ts=t_eval))
        # data[i, j, :, :] = solution.ys

        # ys = RK4(glycolytic_oscilator, 
        #             (t_eval[0], t_eval[-1]),
        #             initial_state,
        #             (selected_params,), 
        #             t_eval=t_eval, 
        #             subdivisions=2)
        # data[i, j, :, :] = ys

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'
elif split == "adapt_test":
  filename = savepath+'adapt_data_test.npz'

np.savez(filename, t=t_eval, X=data)










#%%


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

n_steps_per_traj

# %%
