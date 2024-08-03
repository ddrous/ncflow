
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# import diffrax
# from nodax import RK4
# import jax.numpy as jnp
import numpy as np

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
	parser = argparse.ArgumentParser(description='Brusselator dataset generation script.')
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
res = 8  # 16x16 grid resolution

def laplacian2D(a):
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

    return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (dx ** 2)   ## At the numerator, we are still computing the weighted difference. The sum of stencil values must be 0.

def vec_to_mat(vec_uv, res):
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
    A = np.random.uniform(0.5, 2.0)
    B = np.random.uniform(1.25, 5.0)

    U = A * np.ones((res, res))
    V = (B/A) * np.ones((res, res)) + 0.1 * np.random.randn(res, res)

    return mat_to_vec(U, V, res)




## Define the ODE
def brusselator(t, uv, params):
    A, B, Du, Dv = params['A'], params['B'], params['Du'], params['Dv']
    U, V = vec_to_mat(uv, res)
    deltaU = laplacian2D(U)
    deltaV = laplacian2D(V)
    dUdt = Du*deltaU + A - (B+1)*U + (U**2)*(V)
    dVdt = Dv*deltaV + B*U - (U**2)*(V)
    duvdt = mat_to_vec(dUdt, dVdt, res)
    return duvdt



if split == "train" or split=="test":
  # Training environments
  # As = [0.5, 1., 1.5, 2.]
  # Bs = [1.25, 2., 3.25, 5.]
  As = [0.75, 1., 1.25]
  Bs = [3.25, 3.5, 3.75]
  environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]


elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  ## Adaptation environments
  # As = [0.25, 0.75, 1.25, 1.75, 2.25]
  # Bs = [0.5, 1.5, 2.5, 4.0, 6.0]
  As = [0.625, 0.875, 1.125, 1.375]
  Bs = [3.125, 3.375, 3.625, 3.875]
  environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]



if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

T_final = 10

n_steps_per_traj = int(T_final/.5)
# n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2*res*res))

# Time span for simulation
t_span = (0, T_final)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames
max_seed = np.iinfo(np.int32).max

# print("T eval", t_eval)

for j in range(n_traj_per_env):

    np.random.seed(j if not split in ["test", "adapt_test"] else max_seed - j)
    initial_state = get_init_cond(res)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # print("Initial state", initial_state)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(brusselator, t_span, initial_state, args=(selected_params,), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

        # use diffrax instead, with the DoPri5 integrator
        # solution = diffrax.diffeqsolve(diffrax.ODETerm(gray_scott),
        #                                diffrax.Tsit5(),
        #                                args=(selected_params),
        #                                t0=t_span[0],
        #                                t1=t_span[1],
        #                                dt0=1e-1,
        #                                y0=initial_state,
        #                                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-8),
        #                                saveat=diffrax.SaveAt(ts=t_eval),
        #                                max_steps=4096*1)
        # data[i, j, :, :] = solution.ys
        # print("Stats", solution.stats['num_steps'])

        # ys = RK4(gray_scott, 
        #             (t_eval[0], t_eval[-1]),
        #             initial_state,
        #             *(selected_params,), 
        #             t_eval=t_eval, 
        #             subdivisions=50)
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

## Check if nan or inf in data
if np.isnan(data).any() or np.isinf(data).any():
  print("NaN or Inf in data. Exiting without saving...")
else:
  np.savez(filename, t=t_eval, X=data)




#%%

if _in_ipython_session:
    # Extract and plot the gycolytic oscillator
    print("data min and max", data.min(), data.max())
    solution = data[6, 0, :, :]
    t = t_eval
    U, V = vec_to_mat(solution[0], res)
    # fig, ax = plt.subplots()
    # ax.imshow(U, cmap='hot', interpolation='nearest', origin='lower')
    # ax.set_title('U')
    # plt.show()

    # # Define a function to update the plot for each frame
    # def update(frame):
    #     U, V = vec_to_mat(solution[frame], res)
    #     im.set_array(U)  # Update the image data for the current frame
    #     return im,  # Return the updated artist objects

    # fig, ax = plt.subplots()
    # ax.set_title('Brusselator U')

    # U, V = vec_to_mat(solution[0], res)  # Assuming solution[0] is available for initial setup
    # # im = ax.imshow(U, cmap='gist_ncar', interpolation='bilinear', origin='lower', vmin=0, vmax=solution.max())

    # # Create the animation with the update function
    # ani = FuncAnimation(fig, update, frames=n_steps_per_traj, blit=True)

    # # Save the animation
    # # ani.save(savepath + 'brusselator.gif', writer='imagemagick', fps=10)
    # ani.save(savepath + 'brusselator.mp4', fps=10, writer='ffmpeg')


    ## Pick a random x position and plot the time series
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.random.randint(res**2)
    # plt.plot(t, solution[:, x_pos], label=f'U or V at x={x_pos}')
    for e in range(len(environments)):
        plt.plot(t, data[e, 2, :, x_pos-10], label=f'U or V at x={x_pos} for env {e}')
    # plt.legend()
    plt.title('Brusselator - Single Position')
