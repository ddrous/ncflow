
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
import os


try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False

print("Running this script in ipython (Jupyter) session ?", _in_ipython_session)

## Parse the three arguments from the command line: "train", the foldername, and the seed

import argparse


if _in_ipython_session:
	args = argparse.Namespace(split='adapt', savepath="./data/")
else:
	parser = argparse.ArgumentParser(description='Simple Pendulum')
	parser.add_argument('--split', type=str, help='Generate "train", "test", "adapt", "adapt_test", or "adapt_huge" data', default='train', required=False)
	parser.add_argument('--savepath', type=str, help='Description of optional argument', default='data/', required=False)

	args = parser.parse_args()

split = args.split
assert split in ["train", "test", "adapt", "adapt_test", "adapt_huge"], "Split must be either 'train', 'test', 'adapt', 'adapt_test', 'adapt_huge'"

savepath = args.savepath

if not os.path.exists(savepath):
    os.makedirs(savepath)

print('=== Parsed arguments to generate data ===')
print(' Split:', split)
print(' Savepath:', savepath)
print()



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


if split == "train" or split=="test":
  # Training environments
  environments = [(1., g) for g in list(np.linspace(2, 24, 25))]

elif split == "adapt" or split == "adapt_test":
  ## Adaptation environments
  environments = [(1., 10.25), (1., 14.75)]


if split == "train":
  n_traj_per_env = 12     ## training
elif split == "test" or split == "adapt_test":
  n_traj_per_env = 32     ## testing
elif split == "adapt":
  n_traj_per_env = 1     ## adaptation

n_steps_per_traj = int(10/0.25)

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames
max_seed = np.iinfo(np.int32).max

for j in range(n_traj_per_env):

    np.random.seed(j if not split in ["test", "adapt_test"] else max_seed - j)

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        initial_state_0 = np.random.uniform(-np.pi/3, np.pi/3, (1,))
        initial_state_1 = np.random.uniform(-1, 1, (1,))
        initial_state = np.concatenate([initial_state_0, initial_state_1], axis=0)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(simple_pendulum, t_span, initial_state, args=(selected_params[0], selected_params[1]), t_eval=t_eval)
        data[i, j, :, :] = solution.y.T

# Save t_eval and the solution to a npz file
if split == "train":
  filename = savepath+'train.npz'
elif split == "test":
  filename = savepath+'test.npz'
elif split == "adapt":
  filename = savepath+'oof_train.npz'
elif split == "adapt_test":
  filename = savepath+'ood_test.npz'
elif split == "adapt_huge":
  filename = savepath+'grid_train.npz'

np.savez(filename, t=t_eval, X=data)





if _in_ipython_session:
  # Extract the solution
  angle, velocity = solution.y

  # Create an animation of the Lotka-Volterra system
  fig, ax = plt.subplots()
  eps = 0.5
  ax.set_xlim(np.min(angle)-eps, np.max(angle)+eps)
  ax.set_ylim(np.min(velocity)-eps, np.max(velocity)+eps)
  ax.set_xlabel('Angle')
  ax.set_ylabel('Velocity')

  concentrations, = ax.plot([], [], 'r-', lw=1)
  time_template = 'Time = %.1fs'
  time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

  def animate(i):
      concentrations.set_data(angle[:i], velocity[:i])
      time_text.set_text(time_template % t_eval[i])
      return concentrations, time_text

  ani = FuncAnimation(fig, animate, frames=len(t_eval), interval=5, repeat=False, blit=True)  # Shortened interval
  plt.show()

  ani.save('data/simple_pendulum.gif', fps=30)
