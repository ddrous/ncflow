#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image
import math

import diffrax
from nodax import RK4
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
	args = argparse.Namespace(split='train', savepath="./tmp/", seed=2026, verbose=1)
else:
	parser = argparse.ArgumentParser(description='Gray-Scott dataset generation script.')
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

## Define the ODE
mass, damping, nat_freq = 1, 0.1, 1
def forced_oscillator(t, X, params):
    x, v = X
    dxdt = v
    dvdt = -2 * damping*nat_freq*v - (nat_freq**2)*x + params(t)
    return jnp.array([dxdt, dvdt])

def get_init_cond():
    # x0 = np.random.uniform(-1, 1, (1,))
    # v0 = np.random.uniform(-1, 1, (1,))
    # return np.concatenate([x0, v0])
    return np.random.random(2) + 1.


def sin(t):
    return jnp.sin(t)
def cos(t):
    return jnp.cos(t)
def periodic(t):
    return jnp.sin(t) + jnp.cos(t)
# def exponential_decay(t):
#     return jnp.exp(-t)
# def square_wave(t):
#     return jnp.sign(jnp.sin(t))
# def trianular_wave(t):
#     return jnp.sign(jnp.sin(t)) * jnp.abs(jnp.sin(t))
# def log_sigmoid(t):
#     return 1 / (1 + jnp.exp(jnp.log(t)))
# def log_gaussian_pulse(t):
#     return jnp.exp(-(jnp.log(t)**2)/2)
# def square_wave(t):
#     return jnp.sign(jnp.sin(t))
# def impulse(t):
#     return jnp.where(jnp.abs(t - np.pi) < 0.1, 1, 0)
# def sin_plus_square(t):
#     return jnp.sin(t) + jnp.sign(jnp.sin(t))
# def sin_minus_square(t):
#     return jnp.sin(t) - jnp.sign(jnp.sin(t))
# def sin_times_square(t):
#     return jnp.sin(t) * jnp.sign(jnp.sin(t))
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
  # environments = [(lambda t: 0.1*i*jnp.cos(i*t**2)) for i in range(0, 5)]
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

for j in range(n_traj_per_env):

    # initial_state = get_init_cond()

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        initial_state = get_init_cond()

        # print("Initial state", initial_state)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(forced_oscillator, t_span, initial_state, args=(selected_params,), t_eval=t_eval, method='RK45')
        data[i, j, :, :] = solution.y.T

        # # use diffrax instead, with the DoPri5 integrator
        # solution = diffrax.diffeqsolve(diffrax.ODETerm(gray_scott),
        #                                diffrax.Tsit5(),
        #                                args=(selected_params),
        #                                t0=t_span[0],
        #                                t1=t_span[1],
        #                                dt0=1e-1,
        #                                y0=initial_state,
        #                                stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        #                                saveat=diffrax.SaveAt(ts=t_eval),
        #                                max_steps=4096*1)
        # data[i, j, :, :] = solution.ys
        # print("Stats", solution.stats['num_steps'])

        # ys = RK4(forced_oscillator, 
        #             (t_eval[0], t_eval[-1]),
        #             initial_state,
        #             *(selected_params,), 
        #             t_eval=t_eval, 
        #             subdivisions=100)
        # data[i, j, :, :] = ys






# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'

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
