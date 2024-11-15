#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


## Load data from mape_scores.csv
mses = np.load('ood_crit_all_mse.npy.npz')['arr_0']
mses

mapes = np.load('ood_crit_all_mape.npy.npz')['arr_0']
mapes

coords = [np.array([b, d]) for b in np.linspace(0.25, 1.25, 51) for d in np.linspace(0.25, 1.25, 51)]
coords = np.stack(coords, axis=0)


#%%

## Remove the outliers above 20 in the mape
# df = df[(df['mape'] > 0) & (df['mape'] < 5)]

# Create a pivot table
x, y, z = coords[:,0], coords[:,1], mapes
# x, y, z = coords[:,0], coords[:,1], mses

# x, y, z = x[z<1], y[z<1], z[z<1]
# z = np.log(z)

# Create a grid of points
nb_points = 51
xx = np.linspace(x.min(), x.max(), nb_points)
yy = np.linspace(y.min(), y.max(), nb_points)
xx, yy = np.meshgrid(xx, yy)

# Interpolate the z values at the points in the grid
# zz = griddata((x, y), z, (xx, yy), method='linear')
zz = np.reshape(z, (nb_points, nb_points))

print(zz.shape)

# Plot contourf against x and y coordinates
fig, ax = plt.subplots(figsize=(14, 12))
# ax.contour(xx, yy, zz, levels=[np.log(i) for i in [5, 10, 15, 20]], cmap='grey')
# ax.contour(xx, yy, zz, levels=[0.1, 0.3, 2.15], cmap='grey')
# ax.contour(xx, yy, zz, levels=[0.1,2], cmap='grey')
# c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm')
c = ax.contourf(xx, yy, zz, levels=150, cmap='coolwarm', norm="log")
# c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm', vmin=0, vmax=1)

# ## Set colobar ticks labels
cbar = fig.colorbar(mappable=c, ax=ax)
# cbar.set_ticks([np.log(t+1e-1) for t in cbar.ax.get_yticks()])
# cbar.set_ticklabels([np.exp(t) for t in cbar.ax.get_yticks()])

print(cbar.ax.get_yticks())

# c.cmap.set_over('grey')
# plt.colorbar(c)
# plt.tick_params(axis='y', which='minor')
# from matplotlib.ticker import FormatStrFormatter
# ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))


## Colorbar y label
cbar.set_label('Log MAPE', labelpad=20, fontsize=34)

## Increase font size of colorbar ticks
cbar.ax.tick_params(labelsize=24)

## Place the training data points on the plot with a X
train_envs = [
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
adapt_envs = [
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 1.125},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 1.125},
]


beta_train = [env["beta"] for env in train_envs]
delta_train = [env["delta"] for env in train_envs]
plt.scatter(beta_train, delta_train, c='yellow', marker='o', s=700, label='Training Environments')
## Connect these points with lines in a circular manner
# for i in range(len(beta_train)):
#     plt.plot([beta_train[i], beta_train[(i+1)%len(beta_train)]], [delta_train[i], delta_train[(i+1)%len(beta_train)]], c='yellow')


beta_adapt = [env["beta"] for env in adapt_envs]
delta_adapt = [env["delta"] for env in adapt_envs]
print(beta_adapt)
print(delta_adapt)
plt.scatter(beta_adapt, delta_adapt, c='indigo', marker='+', s=700, label='Adaptation Environments')
## Label the adaptation environments
for i, txt in enumerate(adapt_envs):
    plt.annotate(r"$e_"+str(i)+"$", (beta_adapt[i]-5e-2, delta_adapt[i]+3e-2), fontsize=28, color='indigo')


## Set the x and y labels from 0.25 to 1.25
plt.xticks(np.linspace(0.25, 1.25, 5))
plt.yticks(np.linspace(0.25, 1.25, 5))

plt.xlabel(r'$\beta$', fontsize=24)
plt.ylabel(r'$\delta$', fontsize=24)

## Ticks size as well as labels
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)



# plt.legend()
plt.show()



## Save the figure
# fig.savefig('mapes_contourf.png', dpi=300)
# fig.savefig('mse_contourf.pdf', dpi=300)

fig.savefig('mapes_contourf_log.png', dpi=600, bbox_inches='tight')


#%%
## Do the same thing as above without interpolating
fig, ax = plt.subplots()
c = ax.scatter(x, y, c=z, cmap='coolwarm', s=50)
plt.colorbar(c)

# cbar = fig.colorbar(mappable=c, ax=ax)
# cbar.set_ticks([t for t in cbar.ax.get_yticks()])
# cbar.set_ticklabels([str(np.exp(t))[:2] for t in cbar.ax.get_yticks()])

plt.show()



#%%
## Plot using heatmap
fig, ax = plt.subplots()
ax.contour(xx, yy, zz, levels=[0.1, 0.3, 2.15], cmap='grey', alpha=0.25)
c = ax.imshow(zz, cmap='coolwarm', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.colorbar(c)
plt.show()

# %%

