#%%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


## Load data from mape_scores.csv
mses = np.load('ood_crit_all_mse.npy')
mses

mapes = np.load('ood_crit_all_mape.npy')
mapes

coords = [np.array([b, d]) for b in np.linspace(0.25, 1.25, 51) for d in np.linspace(0.25, 1.25, 51)]
coords = np.stack(coords, axis=0)


#%%

## Remove the outliers above 20 in the mape
# df = df[(df['mape'] > 0) & (df['mape'] < 5)]

# Create a pivot table
x, y, z = coords[:,0], coords[:,1], mapes

# x, y, z = x[z<1], y[z<1], z[z<1]
# z = np.log10(z)

# Create a grid of points
nb_points = 25
xx = np.linspace(x.min(), x.max(), nb_points)
yy = np.linspace(y.min(), y.max(), nb_points)
xx, yy = np.meshgrid(xx, yy)

# Interpolate the z values at the points in the grid
# zz = griddata((x, y), z, (xx, yy), method='linear')
zz = np.reshape(z, (nb_points, nb_points))

## Every all nans with 0
# zz = np.nan_to_num(zz, nan=0)
# print(np.isnan(zz).sum())

print(zz.shape)

# Plot contourf against x and y coordinates
fig, ax = plt.subplots(figsize=(14, 12))
# ax.contour(xx, yy, zz, levels=[np.log(i) for i in [5, 10, 15, 20]], cmap='grey')
# ax.contour(xx, yy, zz, levels=[0.1, 0.3, 2.15], cmap='grey')
# ax.contour(xx, yy, zz, levels=[0.1,2], cmap='grey')
# c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm')

# c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm', norm="linear", extend="min", vmin=0)

c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm', norm="linear", extend='both', vmin=0, vmax=8)
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


## Set colorbar ticks every 4%
mylist = [0, 2, 4, 8, 12, 16]
cbar.set_ticks(mylist)
# cbar.set_ticklabels([str(t) for t in cbar.ax.get_yticks()])
cbar.set_ticklabels([str(t) for t in mylist])

# ## Set colorbar ticks with powers of 10
# mylist = [-6.5, -6, -5.5, -5, -4, -3]
# cbar.set_ticks(mylist)
# # cbar.set_ticklabels([str(t) for t in cbar.ax.get_yticks()])
# cbar.set_ticklabels([f"{10**t:.0e}" for t in mylist])


## Colorbar y label
# cbar.set_label('MSE', labelpad=20, fontsize=34)
cbar.set_label('MAPE (%)', labelpad=20, fontsize=34)

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
## Connect these points with lines to form grid with 9 nodes
plt.plot([0.5, 1.0], [0.5, 0.5], c='yellow', linestyle='--', linewidth=2)
plt.plot([0.5, 1.0], [0.75, 0.75], c='yellow', linestyle='--', linewidth=2)
plt.plot([0.5, 1.0], [1.0, 1.0], c='yellow', linestyle='--', linewidth=2)
plt.plot([0.5, 0.5], [0.5, 1.0], c='yellow', linestyle='--', linewidth=2)
plt.plot([0.75, 0.75], [0.5, 1.0], c='yellow', linestyle='--', linewidth=2)
plt.plot([1.0, 1.0], [0.5, 1.0], c='yellow', linestyle='--', linewidth=2)




beta_adapt = [env["beta"] for env in adapt_envs]
delta_adapt = [env["delta"] for env in adapt_envs]
print(beta_adapt)
print(delta_adapt)
plt.scatter(beta_adapt, delta_adapt, c='indigo', marker='+', s=900, label='Adaptation Environments')
## Label the adaptation environments
for i, txt in enumerate(adapt_envs):
    plt.annotate(r"$e_"+str(i)+"$", (beta_adapt[i]-5e-2, delta_adapt[i]+3e-2), fontsize=28, color='indigo')


## Set the x and y labels from 0.25 to 1.25
plt.xticks(np.linspace(0.25, 1.25, 5))
plt.yticks(np.linspace(0.25, 1.25, 5))

plt.xlabel(r'$\beta$', fontsize=28)
plt.ylabel(r'$\delta$', fontsize=28)

## Ticks size as well as labels
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)



# plt.legend()
plt.show()



## Save the figure
# fig.savefig('mapes_contourf.png', dpi=300)
fig.savefig('mape_contourf.pdf', dpi=300, bbox_inches='tight')

# fig.savefig('mapes_contourf.png', dpi=600, bbox_inches='tight')


#%%

