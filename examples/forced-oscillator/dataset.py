#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import jax.numpy as jnp
import os
import argparse

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

if _in_ipython_session:
	args = argparse.Namespace(split='train', savepath="./data/", verbose=1)
else:
	parser = argparse.ArgumentParser(description='Forced Pendulum dataset generation script.')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='data/', required=False)
	parser.add_argument('--verbose',type=int, help='Whether to print details or not ?', default=1, required=False)

	args = parser.parse_args()

split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"

savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)

if args.verbose != 0:
  print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)
  print('=== Parsed arguments to generate data ===')
  print(' Split:', split)
  print(' Savepath:', savepath)
  print()


#%%

## Define the ODE
mass, damping, nat_freq = 1, 0.1, 1
def forced_oscillator(t, X, params):
    x, v = X
    dxdt = v
    dvdt = -2 * damping*nat_freq*v - (nat_freq**2)*x + params(t)
    return jnp.array([dxdt, dvdt])

def get_init_cond():
    return np.random.random(2) + 1.

def sin(t):
    return jnp.sin(t)
def cos(t):
    return jnp.cos(t)
def periodic(t):
    return jnp.sin(t) + jnp.cos(t)
def expcos(t):
    return jnp.exp(jnp.cos(t))
def sincos(t):
    return jnp.sin(jnp.cos(t))
def expcos(t):
   return jnp.exp(jnp.cos(t))
def sinperiodic(t):
    return jnp.sin(periodic(t))
def sinhperiodic(t):
    return jnp.sinh(periodic(t))
def sinhsin(t):
    return jnp.sinh(jnp.sin(t))
def sinhcos(t):
    return jnp.sinh(jnp.cos(t))

if split == "train" or split=="test":
  # Training environments
  environments = [sin, cos, periodic, expcos, sincos, sinperiodic, sinhperiodic, sinhsin]

elif split == "adapt":
  environments = [sinhcos]

if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 12     ## adaptation

t_span = (0, 6*np.pi)  # Shortened time span
n_steps_per_traj = math.ceil(t_span[-1]/0.1)

# Time span for simulation
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))
max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split=="test" else max_seed - j)
    initial_state = get_init_cond()

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(forced_oscillator, t_span, initial_state, args=(selected_params,), t_eval=t_eval, method='RK45')
        data[i, j, :, :] = solution.y.T

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train.npz'
elif split == "test":
  filename = savepath+'test.npz'
elif split == "adapt":
  filename = savepath+'ood_train.npz'

## Check if nan or inf in data
if np.isnan(data).any() or np.isinf(data).any():
  print("NaN or Inf in data. Exiting without saving...")
else:
  np.savez(filename, t=t_eval, X=data)


#%%

if _in_ipython_session:
    fig, ax = plt.subplots(figsize=(10, 6))

    for env in range(len(environments)):
        x, v = data[env, 0, :, 0], data[env, 0, :, 1]
        ax.plot(t_eval, x, label=f'{env} - {environments[env].__name__}')

    ax.set_xlabel('Time')
    ax.set_ylabel('Displacement')
    ax.set_title('Forced Oscillator (1st trajectory in each env)')
    ax.legend()
    plt.show()

# %%
