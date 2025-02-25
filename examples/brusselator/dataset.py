
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
	parser = argparse.ArgumentParser(description='Brusselator dataset generation script.')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='tmp/', required=False)
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
res = 8

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

    return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (dx ** 2)

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
  As = [0.75, 1., 1.25]
  Bs = [3.25, 3.5, 3.75]
  environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]

elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  ## Adaptation environments
  As = [0.875, 1.125, 1.375]
  Bs = [3.125, 3.375, 3.625, 3.875]
  environments = [{"A": A, "B": B, "Du": 1.0, "Dv": 0.1} for A in As for B in Bs]

if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1      ## adaptation

T_final = 10

n_steps_per_traj = int(T_final/.5)

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2*res*res))

# Time span for simulation
t_span = (0, T_final)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj, endpoint=False)  # Fewer frames
max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split in ["test", "adapt_test"] else max_seed - j)
    initial_state = get_init_cond(res)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(brusselator, t_span, initial_state, args=(selected_params,), t_eval=t_eval)
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
    solution = data[6, 0, :, :]
    t = t_eval
    U, V = vec_to_mat(solution[0], res)

    ## Pick a random x position and plot the time series
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.random.randint(res**2)
    for e in range(len(environments)):
        plt.plot(t, data[e, 2, :, x_pos-10], label=f'U or V at x={x_pos} for env {e}')
    # plt.legend()
    plt.title('Brusselator - Single Position')
