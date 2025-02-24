
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

import diffrax
from ncf import RK4
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
	args = argparse.Namespace(split='train', savepath="tmp/", seed=2026, verbose=1)
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# import jax
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
# import diffrax

# Define the Selkov system: https://en.wikipedia.org/wiki/Hopf_bifurcation
def selkov(t, y, a, b):
    x, y = y
    dx = -x + a*y + (x**2)*y
    dy = b - a*y - (x**2)*y
    return np.array([dx, dy])


if split == "train" or split=="test":
  # Training environments
  # environments = [(0.1, b) for b in np.linspace(-1.0, 1.0, 21)]

  environments_train = [(0.1, b) for b in list(np.linspace(-1, -0.25, 7))\
        + list(np.linspace(-0.1, 0.1, 7))\
        + list(np.linspace(0.25, 1., 7))]

  environments_adapt = [(0.1, b) for b in [-1.25, -0.65, -0.05, 0.02, 0.6, 1.2]]

  environments = environments_train + environments_adapt
  ## Sort the elements in the environments based on the second tuple element
  environments = sorted(environments, key=lambda x: x[1])

  # environments = [(0.1, b) for b in np.linspace(0.2, 1.0, 16)]
  # environments = [(0.1, b) for b in np.linspace(0.5, 0.6, 1)]
elif split == "adapt" or split == "adapt_test":
  ## Adaptation environments
  # environments = [(0.1, b) for b in np.linspace(-1.25, 1.25, 21)[::4]]
  # environments = [(0.1, b) for b in np.linspace(0.1, 1.1, 16)[::4]]
  environments = [(0.1, b) for b in [-1.25, -0.65, -0.05, 0.02, 0.6, 1.2]]



if split == "train":
  n_traj_per_env = 1     ## training
elif split == "test" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 4     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = 1001

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 40)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split not in ["test", "adapt_test", "adapt_huge"] else max_seed - j)
    initial_state = np.random.uniform(0, 3, 2)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        # initial_state = [0, 2]

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(selkov, t_span, initial_state, args=(selected_params[0], selected_params[1]), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T











# %%
if _in_ipython_session:

  n_envs = len(environments)
  # fig, axs = plt.subplots(1, n_envs, figsize=(5*n_envs, 5))
  fig, axs = plt.subplots(3, n_envs//3, figsize=(5*n_envs//3, 5*3), sharex=True, sharey=True)
  axs = axs.flatten()

  print("Total number of environments:", n_envs)

  min_x, max_x = data[:, :, :, 0].min(), data[:, :, :, 0].max()
  min_y, max_y = data[:, :, :, 1].min(), data[:, :, :, 1].max()
  eps = 0.4

  # min_x, max_x = -3, 3
  # min_y, max_y = -3, 3

  for e, (a, b) in enumerate(environments):
      t_span = [0, 1000]
      # sol = solve_ivp(selkov, t_span, y0, args=(a, b), dense_output=True)
      # y = sol.sol(t)

      y = data[e, 0, :, :].T
      color = 'darkblue' if (a,b) in environments_train else 'crimson'
      # plt.plot(y[0], y[1], label=f'b={b:.2f}', color=color)
      # axs[e].plot(y[0], y[1], label=f'b={b:.2f}', color=color)
      axs[e].plot(y[0], y[1], color=color)

      ## Put a cross at the initial condition
      axs[e].plot(y[0,0], y[1,0], 'kx', markersize=10)

      # axs[e].set_title(f'b={b:.2f}')

      axs[e].set_xlim(min_x-eps, max_x+eps)
      axs[e].set_ylim(min_y-eps, max_y+eps)

      leg = axs[e].legend(fontsize=18, loc='upper left')
      # leg.set_title(f'b={b:.2f}', prop={'size':28})
      leg.set_title(r"$b="+str(np.round(b,2))+"$", prop={'size':28})

      ## Set ticks fontsize
      axs[e].tick_params(axis='both', which='major', labelsize=18)

      ## Place x and y axis labels on some axis
      if e %9 == 0:
        axs[e].set_ylabel(r'$y$', fontsize=28)
      if e >= 18:
        axs[e].set_xlabel(r'$x$', fontsize=28)

  plt.tight_layout()

  # plt.xlabel('x')
  # plt.ylabel('y')
  # plt.legend()
  plt.show()


  ## Save figure as png
  fig.savefig(savepath+'selkov_vis.pdf', dpi=300, bbox_inches='tight')












#%%



ind = [0.00050261, 0.00207071, 0.00152377, 0.00131768, 0.00103465, 0.00022663,
 0.00017005, 0.00020905, 0.00026408, 0.00030019, 0.00027802, 0.00018209,
 0.00012141, 0.0002801,  0.00015262, 0.00015415, 0.00044588, 0.00065428,
 0.00119124, 0.00122931, 0.001177]

params_ind = [b for _, b in environments_train]

ood = [0.00047352, 0.00027767, 0.00000721, 0.00000548, 0.0001895,  0.0002639 ]
params_ood = [b for _, b in environments_adapt]

fig = plt.figure(figsize=(10, 4))
plt.plot(params_ind, ind, 'o-', label='In-Domain', color='darkblue', markersize=10)
plt.plot(params_ood, ood, 'o-', label='OOD', color='crimson', lw=3, markersize=10)

## Plot with bars
# plt.bar(params_ind, ind, color='darkblue', label='In-Domain')
# plt.bar(params_ood, ood, color='crimson', label='OOD')

## Set the x ticks to be the b values
plt.xticks(np.linspace(-1.25, 1.25, 15), fontsize=8)

plt.xlabel(r'$\leftarrow b \rightarrow$', fontsize=18)
# plt.ylabel('MSE', fontsize=18)
plt.yscale('log')
plt.legend(fontsize=18)

# plt.tight_layout()

fig.savefig(savepath+'selkov_mse.pdf', dpi=300, bbox_inches='tight')




#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='talk')

from sklearn.manifold import TSNE

contexts_ind = np.load('tmp/contexts_ind.npy')
contexts_ood = np.load('tmp/contexts_ood.npy')

print(contexts_ind.shape, contexts_ood.shape)

## Let's do a t-SNE to visualize the contexts ind and ood in the same plot

tsne = TSNE(n_components=2, random_state=42, perplexity=26)
X = np.vstack([contexts_ind, contexts_ood])
X_embedded = tsne.fit_transform(X)

fig = plt.figure(figsize=(6, 5))
plt.scatter(X_embedded[:len(contexts_ind), 0], X_embedded[:len(contexts_ind), 1], label='In-Domain', color='darkblue')
plt.scatter(X_embedded[len(contexts_ind):, 0], X_embedded[len(contexts_ind):, 1], label='OOD', color='crimson')

plt.legend(fontsize=18)
plt.axis('off')

fig.savefig('tmp/selkov_tsne.pdf', dpi=300, bbox_inches='tight')


