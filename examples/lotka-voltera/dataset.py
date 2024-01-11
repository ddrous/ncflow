
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image

## Set numpy seed for reproducibility
np.random.seed(5)


#%%

Image(filename="tmp/coda_dataset.png")


#%%


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

import jax
# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import diffrax

# Define the Lotka-Volterra system
def lotka_volterra(t, state, alpha, beta, delta, gamma):
    x, y = state
    dx_dt = alpha * x - beta * x * y
    dy_dt = delta * x * y - gamma * y
    return [dx_dt, dy_dt]


## Training environments
environments = [
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
    {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
    {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
    {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},
]

# ## Adaptation environments
# environments = [
#     {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
#     {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 1.125},
#     {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
#     {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 1.125},
# ]

n_traj_per_env = 4     ## training
# n_traj_per_env = 32     ## testing
# n_traj_per_env = 1     ## adaptation

# n_steps_per_traj = int(10/0.5)    ## from coda
n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

# Time span for simulation
t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

for j in range(n_traj_per_env):

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Initial conditions (prey and predator concentrations)
        initial_state = np.random.uniform(1, 3, (2,))

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(lotka_volterra, t_span, initial_state, args=(selected_params["alpha"], selected_params["beta"], selected_params["delta"], selected_params["gamma"]), t_eval=t_eval)

        data[i, j, :, :] = solution.y.T

# Extract the solution
prey_concentration, predator_concentration = solution.y
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
    time_text.set_text(time_template % solution.t[i])
    return concentrations, time_text

ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=5, repeat=False, blit=True)  # Shortened interval
plt.show()

# Save t_eval and the solution to a npz file
np.savez('tmp/dataset_big.npz', t=solution.t, X=data)

## Save the movie to a small mp4 file
ani.save('tmp/lotka_volterra.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
