
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import torch
from torch.utils.data import Dataset
import math
import shelve

try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	# args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	args = argparse.Namespace(split='adapt_test', savepath="./tmp/", seed=2026, verbose=1)
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

res = 32

if split == "train" or split=="test":
# Training environments
  tt = torch.linspace(0, 1, res + 1)[0:-1]
  X, Y = torch.meshgrid(tt, tt)
  environments = [
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 8e-4},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 9e-4},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.0e-3},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.1e-3},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.2e-3},
        ]   


elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  ## Adaptation environments
  tt = torch.linspace(0, 1, res + 1)[0:-1]
  X, Y = torch.meshgrid(tt, tt)
  f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
  viscs = [8.5e-4, 9.5e-4, 1.05e-3, 1.15e-3]
  environments = [{"f": f, "visc": visc} for visc in viscs]


if split == "train":
  n_traj_per_env = 16     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation







class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NavierStokesDataset(Dataset):
    def __init__(self, n_data_per_env, size, t_horizon, params, dt_eval, dx=2., buffer_file=None, method='RK45', group='train'):
        super().__init__()
        self.size = int(size)  # size of the 2D grid
        self.params_eq = params
        self.forcing_zero = params[0]['f']
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.dx = dx  # space step discretized domain [-1, 1]
        self.t_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt_eval)  # number of iterations
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt_eval = dt_eval
        self.dt = 1e-3
        self.buffer = shelve.open(buffer_file)
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.method = method
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                         torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t

    def _get_init_cond(self, index):
        torch.manual_seed(index if not self.test else self.max - index)
        if self.buffer.get(f'init_cond_{index}') is None:
            w0 = self.sampler.sample()
            state, _ = self.navier_stokes_2d(w0, f=self.forcing_zero, visc=8e-4, T=30.0,
                                             delta_t=self.dt, record_steps=20)
            init_cond = state[:, :, -1, 0]
            self.buffer[f'init_cond_{index}'] = init_cond.numpy()
        else:
            init_cond = torch.from_numpy(self.buffer[f'init_cond_{index}'])

        return init_cond

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt_eval).float()
        # if self.buffer.get(f'{env},{env_index}') is None:
        if True:
            print(f'calculating index {env_index} of env {env}')
            w0 = self._get_init_cond(env_index)
            
            # w0 = F.interpolate(w0.unsqueeze(0).unsqueeze(0), scale_factor=2).squeeze(0).squeeze(0)
            state, _ = self.navier_stokes_2d(w0, f=self.params_eq[env]['f'], visc=self.params_eq[env]['visc'],
                                             T=self.t_horizon, delta_t=self.dt, record_steps=self.n)
            # h, w, t, nc
            state = state.permute(3, 2, 0, 1)[:, :self.n]  # nc, t, h, w
            # state = F.avg_pool2d(state, kernel_size=2, stride=2)
            # print(state.shape)
            self.buffer[f'{env},{env_index}'] = {'state': state.numpy()}
            return {'state': state, 't': t, 'env': env, 'index': env_index}
        else:
            buf = self.buffer[f'{env},{env_index}']
            return {'state': torch.from_numpy(buf['state'][:, :self.n]), 't': t, 'env': env, 'index': index}

    def __len__(self):
        return self.len






T_final = 10
n_steps_per_traj = int(T_final/1)
t_eval = np.linspace(0, T_final, n_steps_per_traj, endpoint=False)
data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 1*res*res))

dataset_train_params = {
    "n_data_per_env": n_traj_per_env, "t_horizon": T_final, "dt_eval": 1, "method": "euler", "size": res, "group": "train",
    "buffer_file": f"{savepath}/ns_buffer_train_3env_08-12_32.shelve",  # ns_buffer_train_30+10_1e-3.shelve
    "params": environments
}

dataset = NavierStokesDataset(**dataset_train_params)

for batch in dataset:
    # print(batch['state'].shape)
    data[batch['env'], batch['index'], :, :] = batch['state'].reshape(n_steps_per_traj, 1*res*res)
    # break







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
    solution = data[0, 0, :, :]
    t = t_eval
    U = solution[0].reshape(res, res)
    fig, ax = plt.subplots()
    ax.imshow(U, cmap='hot', interpolation='nearest', origin='lower')
    ax.set_title('U')
    plt.show()

    # Define a function to update the plot for each frame
    def update(frame):
        U = solution[frame].reshape(res, res)
        im.set_array(U)  # Update the image data for the current frame
        return im,  # Return the updated artist objects

    fig, ax = plt.subplots()
    ax.set_title('Navier-Stokes')

    U = solution[0].reshape(res, res)  # Assuming solution[0] is available for initial setup
    im = ax.imshow(U, cmap='gist_ncar', interpolation='bilinear', origin='lower')

    # Create the animation with the update function
    ani = FuncAnimation(fig, update, frames=n_steps_per_traj, blit=True)

    # Save the animation
    ani.save(savepath + 'navier-stokes.gif', writer='imagemagick', fps=10)

# %%
