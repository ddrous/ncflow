#%%
import pandas as pd
import numpy as np
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
z = np.log(z)

# Create a grid of points
nb_points = 510
xx = np.linspace(x.min(), x.max(), nb_points)
yy = np.linspace(y.min(), y.max(), nb_points)
xx, yy = np.meshgrid(xx, yy)

# Interpolate the z values at the points in the grid
zz = griddata((x, y), z, (xx, yy), method='linear')

print(zz.shape)

# Plot contourf against x and y coordinates
fig, ax = plt.subplots()
# ax.contour(xx, yy, zz, levels=[np.log(i) for i in [5, 10, 15, 20]], cmap='grey')
# ax.contour(xx, yy, zz, levels=10, cmap='grey')
c = ax.contourf(xx, yy, zz, levels=500, cmap='coolwarm')
plt.colorbar(c)
plt.show()

## Save the figure
fig.savefig('mapes_contourf.png', dpi=300)
# fig.savefig('mse_contourf.pdf', dpi=300)

#%%
## Do the same thing as above without interpolating
fig, ax = plt.subplots()
c = ax.scatter(x, y, c=z, cmap='coolwarm', s=50)
plt.colorbar(c)
plt.show()



#%%
## Plot using heatmap
fig, ax = plt.subplots()
c = ax.imshow(zz, cmap='coolwarm', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', interpolation='bilinear')
plt.colorbar(c)
plt.show()

# %%

