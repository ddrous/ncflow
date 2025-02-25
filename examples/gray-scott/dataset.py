
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

import argparse

if _in_ipython_session:
	args = argparse.Namespace(split='train', savepath="./data/", verbose=1)
else:
	parser = argparse.ArgumentParser(description='Gray-Scott dataset generation script.')
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
dx = 1
res = 32  # 32x32 grid resolution

def laplacian2D(a):     ## From https://github.com/yuan-yin/CoDA
    # a_nn | a_nz | a_np
    # a_zn | a    | a_zp
    # a_pn | a_pz | a_pp
    a_zz = a

    a_nz = np.roll(a_zz, (+1, 0), (0, 1))
    a_pz = np.roll(a_zz, (-1, 0), (0, 1))
    a_zn = np.roll(a_zz, (0, +1), (0, 1))
    a_zp = np.roll(a_zz, (0, -1), (0, 1))

    a_nn = np.roll(a_zz, (+1, +1), (0, 1))
    a_np = np.roll(a_zz, (+1, -1), (0, 1))
    a_pn = np.roll(a_zz, (-1, +1), (0, 1))
    a_pp = np.roll(a_zz, (-1, -1), (0, 1))

    return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (dx ** 2)   ## At the numerator, we are still computing the weighted difference.

def vec_to_mat(vec_uv, res=32):
    UV = np.split(vec_uv, 2)
    U = np.reshape(UV[0], (res, res))
    V = np.reshape(UV[1], (res, res))
    return U, V

def mat_to_vec(mat_U, mat_V, res):
    dudt = np.reshape(mat_U, res * res)
    dvdt = np.reshape(mat_V, res * res)
    return np.concatenate((dudt, dvdt))

## Define initial conditions
def get_init_cond(res):
    U = 0.95 * np.ones((res, res))
    V = 0.05 * np.ones((res, res))
    n_block = 3
    for _ in range(n_block):
        r = int(res / 10)
        N2 = np.random.randint(low=0, high=res-r, size=2)
        U[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 0.
        V[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 1.
    return mat_to_vec(U, V, res)


## Define the ODE
def gray_scott(t, uv, params):
    U, V = vec_to_mat(uv, res)
    deltaU = laplacian2D(U)
    deltaV = laplacian2D(V)
    dUdt = (params['r_u'] * deltaU - U * (V ** 2) + params['f'] * (1. - U))
    dVdt = (params['r_v'] * deltaV + U * (V ** 2) - (params['f'] + params['k']) * V)
    duvdt = mat_to_vec(dUdt, dVdt, res)

    return duvdt



if split == "train" or split=="test":
  # Training environments
  environments = [
      {"f": 0.03, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.039, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.03, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
      {"f": 0.039, "k": 0.062, "r_u": 0.2097, "r_v": 0.105}
  ]
elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  ## Adaptation environments
	from itertools import product
	f = [0.033, 0.036]
	k = [0.059, 0.061]
	environments = [{"f": f_i, "k": k_i, "r_u": 0.2097, "r_v": 0.105} for f_i, k_i in product(f, k)]



if split == "train":
  n_traj_per_env = 1     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(400/40)

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2*res*res))

# Time span for simulation
t_span = (0, 400)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames
max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split in ["test", "adapt_test"] else max_seed - j)
    initial_state = get_init_cond(res)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(gray_scott, t_span, initial_state, args=(selected_params,), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train.npz'
elif split == "test":
  filename = savepath+'test.npz'
elif split == "adapt":
  filename = savepath+'ood_train.npz'
elif split == "adapt_test":
  filename = savepath+'ood_test.npz'

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

    ani = FuncAnimation(fig, update, frames=n_steps_per_traj, blit=True)

    # Save the animation
    ani.save(savepath + 'gray_scott.gif', writer='imagemagick', fps=10)

# %%
