
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
	# args = argparse.Namespace(split='train', savepath='tmp/', seed=42)
	args = argparse.Namespace(split='test', savepath="./runs/24012024-084802/", seed=3422)
else:
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='tmp/', required=False)
	parser.add_argument('--seed',type=int, help='Seed to gnerate the data', default=42, required=False)

	args = parser.parse_args()


split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"

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
def simple_pendulum(t, state, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return np.array([theta_dot, theta_ddot])


def rk4_integrator(rhs, y0, t):
  def step(state, t):
    y_prev, t_prev = state
    h = t - t_prev
    k1 = h * rhs(y_prev, t_prev)
    k2 = h * rhs(y_prev + k1/2., t_prev + h/2.)
    k3 = h * rhs(y_prev + k2/2., t_prev + h/2.)
    k4 = h * rhs(y_prev + k3, t + h)
    y = y_prev + 1./6 * (k1 + 2 * k2 + 2 * k3 + k4)
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
  environments = [(1., g) for g in list(np.linspace(2, 24, 100))]

elif split == "adapt" or split == "adapt_test":
  ## Adaptation environments
  environments = [(1., g) for g in list(np.linspace(1, 2, 10))] + [(1., g) for g in list(np.linspace(24, 30, 10))]

if split == "train":
  n_traj_per_env = 4     ## training
elif split == "test":
  n_traj_per_env = 32     ## testing
elif split == "adapt" or split == "adapt_test" or split == "adapt_huge":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(10/0.5)
# n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

for j in range(n_traj_per_env):

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        initial_state_0 = np.random.uniform(-np.pi/3, np.pi/3, (1,))
        initial_state_1 = np.random.uniform(-1, 1, (1,))
        initial_state = np.concatenate([initial_state_0, initial_state_1], axis=0)

        # Solve the ODEs using SciPy's solve_ivp
        # solution = solve_ivp(simple_pendulum, t_span, initial_state, args=(selected_params[0], selected_params[1]), t_eval=t_eval)
        # data[i, j, :, :] = solution.y.T

        rhs = lambda x, t: simple_pendulum(t, x, selected_params[0], selected_params[1])
        solution = rk4_integrator(rhs, initial_state, t_eval)
        data[i, j, :, :] = solution

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train_data.npz'
elif split == "test":
  filename = savepath+'test_data.npz'
elif split == "adapt":
  filename = savepath+'adapt_data.npz'
elif split == "adapt_test":
  filename = savepath+'adapt_test_data.npz'
elif split == "adapt_huge":
  filename = savepath+'adapt_huge_data.npz'

np.savez(filename, t=t_eval, X=data)





# ## Randmly pick a trajectory and plot it, then save it
# e, traj_id = np.random.randint(0, len(environments)), np.random.randint(0, n_traj_per_env)
# print("Plotting environment", e, "trajectory", traj_id)
# traj = data[e, traj_id, :, :]
# plt.plot(t_eval, traj[:, 0], label='theta')
# plt.plot(t_eval, traj[:, 1], label='theta_dot')
# plt.legend()
# plt.savefig('tmp/pendulum.png')
# plt.show()








if _in_ipython_session:
  # Extract the solution
  prey_concentration, predator_concentration = solution.T
  # prey_concentration, predator_concentration = solution.y
  # prey_concentration, predator_concentration = solution.ys

  # Create an animation of the Lotka-Volterra system
  fig, ax = plt.subplots()
  eps = 0.5
  ax.set_xlim(-eps, np.max(prey_concentration)+eps)
  ax.set_ylim(-eps, np.max(predator_concentration)+eps)
  ax.set_xlabel('Preys')
  ax.set_ylabel('Predators')

  concentrations, = ax.plot([], [], 'r-', lw=1, label='Concentrations')
  time_template = 'Time = %.1fs'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  # Add legend
  ax.legend()

  def animate(i):
      concentrations.set_data(prey_concentration[:i], predator_concentration[:i])
      time_text.set_text(time_template % t_eval[i])
      return concentrations, time_text

  ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=5, repeat=False, blit=True)  # Shortened interval
  plt.show()


  ## Save the movie to a small mp4 file
  # ani.save('tmp/lotka_volterra.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
  ani.save('tmp/lotka_volterra.gif', fps=30)

