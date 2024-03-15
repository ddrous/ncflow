
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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

dx = 1
res = 32  # 32x32 grid resolution

def laplacian2D(a):
    # a_nn | a_nz | a_np
    # a_zn | a    | a_zp
    # a_pn | a_pz | a_pp
    a_zz = a

    a_nz = jnp.roll(a_zz, (+1, 0), (0, 1))
    a_pz = jnp.roll(a_zz, (-1, 0), (0, 1))
    a_zn = jnp.roll(a_zz, (0, +1), (0, 1))
    a_zp = jnp.roll(a_zz, (0, -1), (0, 1))

    a_nn = jnp.roll(a_zz, (+1, +1), (0, 1))
    a_np = jnp.roll(a_zz, (+1, -1), (0, 1))
    a_pn = jnp.roll(a_zz, (-1, +1), (0, 1))
    a_pp = jnp.roll(a_zz, (-1, -1), (0, 1))

    return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (dx ** 2)   ## At the numerator, we are still computing the weighted difference. The sum of stencil values must be 0.

def vec_to_mat(vec_uv, res=32):
    UV = jnp.split(vec_uv, 2)
    U = jnp.reshape(UV[0], (res, res))
    V = jnp.reshape(UV[1], (res, res))
    return U, V

def mat_to_vec(mat_U, mat_V, res):
    dudt = jnp.reshape(mat_U, res * res)
    dvdt = jnp.reshape(mat_V, res * res)
    return jnp.concatenate((dudt, dvdt))

## Define initial conditions
def get_init_cond(res):
    # np.random.seed(index if not self.test else self.max - index)
    U = 0.95 * np.ones((res, res))
    V = 0.05 * np.ones((res, res))
    n_block = 3
    for _ in range(n_block):
        r = int(res / 10)
        N2 = np.random.randint(low=0, high=res-r, size=2)
        U[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 0.
        V[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 1.
    # return U, V
    return mat_to_vec(U, V, res)




## Define the ODE
def gray_scott(t, uv, params):
    U, V = vec_to_mat(uv, res)
    deltaU = laplacian2D(U)
    deltaV = laplacian2D(V)
    dUdt = (params['r_u'] * deltaU - U * (V ** 2) + params['f'] * (1. - U))
    dVdt = (params['r_v'] * deltaV + U * (V ** 2) - (params['f'] + params['k']) * V)
    duvdt = mat_to_vec(dUdt, dVdt, res)

    ## Do NaN to NuM
    duvdt = jnp.nan_to_num(duvdt)
    return duvdt



if split == "train" or split=="test":
  # Training environments
  environments = [
      {"f": 0.03, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.039, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.03, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.039, "k": 0.062, "r_u": 0.2097, "r_v": 0.105}
  ]



elif split == "adapt":
  ## Adaptation environments
	from itertools import product
	# f = [0.033, 0.036]
	# k = [0.059, 0.061]
	f = [0.033]
	k = [0.059]
	environments = [{"f": f_i, "k": k_i, "r_u": 0.2097, "r_v": 0.105} for f_i, k_i in product(f, k)]



if split == "train":
  n_traj_per_env = 1     ## training
elif split == "test":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(400/40)
# n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2*res*res))

# Time span for simulation
t_span = (0, 400)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames

for j in range(n_traj_per_env):

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        initial_state = get_init_cond(res)

        # print("Initial state", initial_state)

        # Solve the ODEs using SciPy's solve_ivp
        # solution = solve_ivp(gray_scott, t_span, initial_state, args=(selected_params,), t_eval=t_eval)
        # data[i, j, :, :] = solution.y.T

        # use diffrax instead, with the DoPri5 integrator
        solution = diffrax.diffeqsolve(diffrax.ODETerm(gray_scott),
                                       diffrax.Tsit5(),
                                       args=(selected_params),
                                       t0=t_span[0],
                                       t1=t_span[1],
                                       dt0=1e-1,
                                       y0=initial_state,
                                       stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
                                       saveat=diffrax.SaveAt(ts=t_eval),
                                       max_steps=4096*1)
        data[i, j, :, :] = solution.ys
        # print("Stats", solution.stats['num_steps'])

        # ys = RK4(gray_scott, 
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
    # Extract and plot the gycolytic oscillator
    print("data min and max", data.min(), data.max())
    solution = data[0, 0, :, :]
    t = t_eval
    U, V = vec_to_mat(solution[0], res)
    fig, ax = plt.subplots()
    ax.imshow(U, cmap='hot', interpolation='nearest', origin='lower')
    ax.set_title('U')
    plt.show()

    # Define a function to update the plot for each frame
    def update(frame):
        U, V = vec_to_mat(solution[frame], res)
        im.set_array(U)  # Update the image data for the current frame
        return im,  # Return the updated artist objects

    fig, ax = plt.subplots()
    ax.set_title('Gray-Scott U')

    U, V = vec_to_mat(solution[0], res)  # Assuming solution[0] is available for initial setup
    im = ax.imshow(U, cmap='gist_ncar', interpolation='bilinear', origin='lower')

    # Create the animation with the update function
    ani = FuncAnimation(fig, update, frames=n_steps_per_traj, blit=True)

    # Save the animation
    ani.save(savepath + 'gray_scott.gif', writer='imagemagick', fps=10)

# %%
