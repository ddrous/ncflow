
# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation


## Set numpy seed for reproducibility
np.random.seed(5)


##### Generatate data for multiple simple environemnts


def simple_pendulum(t, state, L, g):
    theta, theta_dot = state
    theta_ddot = -(g / L) * np.sin(theta)
    return [theta_dot, theta_ddot]


nb_envs = 9
environments = []
gs = np.linspace(3, 25, nb_envs)
for e in range(nb_envs):
    # L = np.random.uniform(0.5, 1.5)
    # L = 1
    L = 0.75
    # g = np.random.uniform(3.72, 240.79)
    g = gs[e]
    environments.append({"L": L, "g": g})

for e in range(nb_envs):
    # L = np.random.uniform(0.5, 1.5)
    # L = 1
    L = 1.25
    # g = np.random.uniform(3.72, 240.79)
    g = gs[e]
    environments.append({"L": L, "g": g})


# environments = [
#     {"L": 0.5, "g": 9.81},    ## Eearth
#     {"L": 1.0, "g": 9.81},
#     # {"L": 1.5, "g": 9.81},
#     {"L": 0.5, "g": 24.79},      ## Jupiter
#     {"L": 1.0, "g": 24.79},
#     # {"L": 1.5, "g": 24.79},
#     {"L": 0.5, "g": 3.72},      ## Mars
#     {"L": 1.0, "g": 3.72},
#     # {"L": 1.5, "g": 3.72},
# ]



# # environments = [
# #     {"L": 0.1, "g": 9.81},    ## Eearth
# #     {"L": 1.0, "g": 9.81},
# #     {"L": 10, "g": 9.81},
# #     {"L": 0.1, "g": 24.79},      ## Jupiter
# #     {"L": 1.0, "g": 24.79},
# #     {"L": 10, "g": 24.79},
# #     {"L": 0.1, "g": 3.72},      ## Mars
# #     {"L": 1.0, "g": 3.72},
# #     {"L": 10, "g": 3.72},
# # ]


n_traj_per_env = 128*2      ## Big
# n_traj_per_env = 8      ## Small
n_steps_per_traj = 201

data = np.zeros((len(environments), n_traj_per_env, n_steps_per_traj, 2))

t_span = (0, 10)  # Shortened time span
t_eval = np.linspace(t_span[0], t_span[-1], n_steps_per_traj)  # Fewer frames

for j in range(n_traj_per_env):
    # Initial conditions (prey and predator concentrations)
    initial_state = np.concatenate([np.random.uniform(-np.pi/2, np.pi/2, size=(1,)), 
    # initial_state = np.concatenate([np.array([-np.pi/2]), 
                                    np.random.uniform(-1, 1, size=(1,))])
                                    # np.array([-1])])

    for i, selected_params in enumerate(environments):
        # print("Environment", i)

        # Solve the ODEs using SciPy's solve_ivp
        solution = solve_ivp(simple_pendulum, t_span, initial_state, args=(selected_params['L'], selected_params['g']), t_eval=t_eval)

        data[i, j, :, :] = solution.y.T

# Extract the solution
theta, theta_dot = solution.y

# Create an animation of the pendulum's motion
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_xlim(-2.1, 2.1)
ax.set_ylim(-1.4, 0.25)

pendulum, = ax.plot([], [], 'ro-', lw=2)
time_template = 'Time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    x = [0, selected_params['L'] * np.sin(theta[i])]
    y = [0, -selected_params['L'] * np.cos(theta[i])]
    pendulum.set_data(x, y)
    time_text.set_text(time_template % solution.t[i])
    return pendulum, time_text

ani = FuncAnimation(fig, animate, frames=len(solution.t), interval=10, repeat=True, blit=True)
plt.show()

# ani.save('data/simple_pen.mp4', writer='ffmpeg')

# Save t_eval and the solution to a npz file
np.savez('tmp/simple_pendulum_big.npz', t=solution.t, X=data)
