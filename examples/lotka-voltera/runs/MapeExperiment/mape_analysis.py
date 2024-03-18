#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


## Load data from mape_scores.csv
df = pd.read_csv('mape_scores.csv')
df # columns = beta, delta, and mape

## Remove the outliers above 20 in the mape
df = df[(df['mape'] > 0) & (df['mape'] < 5)]

# Create a pivot table
x, y, z = df['beta'], df['delta'], df['mape']

# Create a grid of points
nb_points = 1000
xx = np.linspace(x.min(), x.max(), nb_points)
yy = np.linspace(y.min(), y.max(), nb_points)
xx, yy = np.meshgrid(xx, yy)

# Interpolate the z values at the points in the grid
zz = griddata((x, y), z, (xx, yy), method='linear')

print(zz.shape)

# Plot contourf against x and y coordinates
fig, ax = plt.subplots()
c = ax.contourf(xx, yy, zz, 100, cmap='viridis')
plt.colorbar(c)
plt.show()




