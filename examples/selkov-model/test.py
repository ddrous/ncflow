#%%

## Simulate thetwo bassins of the spiking neuron model
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import Image
#%%

# ## DEFINE THE ODE
# # def neuron(t, y, a, b, c, d, I):
# #     v, u = y
# #     dv = 0.04 * v**2 + 5*v + 140 - u + I
# #     du = a * (b*v - u)
# #     return [dv, du]

# ## Define the Hodgkin-Huxley model
# def hodgkin_huxley(t, y, I):
#     v, m, h, n = y
#     dv = (I - 120*m**3*h*(v-50) - 36*n**4*(v+77) - 0.3*(v+54.387))/0.01
#     dm = 0.1*(25-v)/(np.exp((25-v)/10) - 1) - 4*np.exp(v/18)
#     dh = 0.07*np.exp(v/20) - 1/(np.exp((30-v)/10) + 1)
#     dn = 0.01*(10-v)/(np.exp((10-v)/10) - 1) - 0.125*np.exp(v/80)
#     return [dv, dm, dh, dn]


# ## SIMULATE THE NEURON
# # a = 0.02
# # b = 0.2
# # c = -65
# # d = 8
# # I = 10
# # y0 = [-70, -14]
# # t_span = [0, 10]
# # sol = solve_ivp(neuron, t_span, y0, args=(a, b, c, d, I), dense_output=True)


# I = 10
# y0 = [-65, 0.05, 0.6, 0.32]
# t_span = [0, 100]
# sol = solve_ivp(hodgkin_huxley, t_span, y0, args=(I,), dense_output=True)



# ## Plot the results
# fig, ax = plt.subplots()
# t = np.linspace(*t_span, 1000)
# y = sol.sol(t)
# ax.plot(t, y[0])
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Membrane potential (mV)')
# plt.show()

# # %%



# ## Now let's do the FitzHugh-Nagumo model
# def fitzhugh_nagumo(t, y, a, b, I):
#     v, w = y
#     dv = -v**3 /3. + v - w + 0.1*I
#     dw = (v + a - b*w) / 12.5
#     return [dv, dw]


# a = 0.7
# b = 2.0
# I = 0.5
# y0 = [-1, 1]
# t_span = [0, 100]
# # sol = solve_ivp(fitzhugh_nagumo, t_span, y0, args=(a, b, I), dense_output=True)

# fig, ax = plt.subplots()
# for I in np.linspace(3.5, 5.45, 15):
#     sol = solve_ivp(fitzhugh_nagumo, t_span, y0, args=(a, b, I), dense_output=True)
#     y = sol.sol(t)
#     ax.plot(y[0], y[1], label=f'I={I:.2f}')
# ax.legend()
# ax.set_xlabel('Membrane potential')
# ax.set_ylabel('Recovery variable')
# plt.show() 


# # ## Plot the results
# # fig, ax = plt.subplots()
# # t = np.linspace(*t_span, 1000)
# # y = sol.sol(t)
# # ax.plot(t, y[0])
# # ax.set_xlabel('Time')
# # ax.set_ylabel('Membrane potential')
# # plt.show()


# # ## Plot sphase space
# # fig, ax = plt.subplots()
# # ax.plot(y[0], y[1])
# # ax.set_xlabel('Membrane potential')
# # ax.set_ylabel('Recovery variable')
# # plt.show()



# # %%

# ## A famous dynamical system with both state fixed point and a limit cycle
# def van_der_pol(t, y, mu):
#     x, y = y
#     dx = y
#     dy = mu*(1-x**2)*y - x
#     return [dx, dy]

# mu = 1
# y0 = [1, 1]
# t_span = [0, 100]
# sol = solve_ivp(van_der_pol, t_span, y0, args=(mu,), dense_output=True)

# # ## Plot the results
# # fig, ax = plt.subplots()
# # t = np.linspace(*t_span, 1000)
# # y = sol.sol(t)
# # ax.plot(t, y[0])
# # ax.set_xlabel('Time')
# # ax.set_ylabel('x')
# # plt.show()


# ## Plot two distinct trajectories that end at both attractors
# fig, ax = plt.subplots()
# t = np.linspace(*t_span, 1000)
# y0 = [1, 1]
# sol = solve_ivp(van_der_pol, t_span, y0, args=(mu,), dense_output=True)
# y = sol.sol(t)
# ax.plot(y[0], y[1], '.')
# # y0 = [-1, -1]
# y0 = [0.05, 0.05]
# sol = solve_ivp(van_der_pol, t_span, y0, args=(mu,), dense_output=True)
# y = sol.sol(t)
# ax.plot(y[0], y[1], "-")
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# plt.show()



# %%

## Selkov model with b varying from 0.25 to 1.25 and a fixed at 1
def selkov(t, y, a, b):
    x, y = y
    dx = -x + a*y + (x**2)*y
    dy = b - a*y - (x**2)*y
    return [dx, dy]

a = .1
y0 = [0, 2]
# y0 = [0.3, 0.5]

# pick 15 values of b. 3 of them with stable fixed points, 9 with limits cycles, and 3 more with fixed points
# for b in np.linspace(0.1, 1.2, 16)[:]:
# for b in np.linspace(-1.0, 1.0, 11)[3:7]:
# for b in np.random.normal(0, 0.2, 15):
for b in list(np.linspace(-1, -0.25, 4))\
        + list(np.linspace(-0.1, 0.1, 7))\
        + list(np.linspace(0.25, 1., 4)):

# for b in np.linspace(0.9, 2.2, 8)[:8]:

# for b in np.linspace(0.2, 1.0, 16)[:4]:
# for b in np.linspace(0.3, 0.9, 15)[0:3]:
# for b in [0.35, 0.55, 0.65, 0.85]:
    t_span = [0, 40]
    t = np.linspace(*t_span, 10000)
    sol = solve_ivp(selkov, t_span, y0, args=(a, b), dense_output=True)
    y = sol.sol(t)
    plt.plot(y[0], y[1], label=f'b={b:.2f}')
    # plt.plot(t, y[0], label=f'b={b:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


print(y[:, -5:])