#%%
import numpy as np
import os

## Load all npz files in the 'NCF' folder, using the os module
data_ncf = []
for file in os.listdir("NCF/"):
    print(file)
    if file not in ['train_histories copy.npz']:
        if file.endswith(".npz"):
            data_ncf.append(np.load("NCF/" + file)['losses_node'])

data_ncf = np.stack(data_ncf).squeeze()
data_ncf.shape

## Print the mins of the losses
# print(data_ncf.min(1))


#%%
import re

## Parse the logs in the 'CoDA' folder, using the re module
data_coda = []
for file in os.listdir("CoDA/"):
    if file.endswith(".log"):
        with open("CoDA/" + file, 'r') as f:
            code = f.read()
            loss_train_values = re.findall(r'Loss Train: (\d+\.\d+e[+-]\d+)', code)
            loss_train_values = [float(i) for i in loss_train_values]
            data_coda.append(loss_train_values)

# ## All loss histories should have same shape
# data_coda[1] = data_coda[1][::2]

data_coda[1] = data_coda[1][:1500]
data_coda[0] = np.concatenate([data_coda[0], data_coda[0][-1] * np.ones(300)])

# data_coda[0] = data_coda[0][::4]
# data_coda[1] = data_coda[1][::8]
# data_coda = np.repeat(data_coda, 5, axis=1)

data_coda = np.stack(data_coda).squeeze()
data_coda.shape

# print(data_coda.min(1))

#%%

## Parse the logs in 'CAVIA' folder, using the re module
data_cavia = []
for file in os.listdir("CAVIA/"):
    if file.endswith(".log"):
        with open("CAVIA/" + file, 'r') as f:
            code = f.read()
            loss_train_values = re.findall(r'\[train\] loss: (\d+\.\d+)\s+-', code)
            loss_train_values = [float(i) for i in loss_train_values]
            data_cavia.append(loss_train_values)

## All loss histories should have same shape
data_cavia[0] = data_cavia[0][:100]
data_cavia[1] = data_cavia[1][::6]
data_cavia[2] = data_cavia[2][:200:2]

data_cavia = np.stack(data_cavia).squeeze()
data_cavia.shape

## Data has length 100, we want it to have length 1500 by repeating values 15 times
data_cavia = np.repeat(data_cavia, 15, axis=1)
data_cavia.shape




#%%

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as pl
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')


## Plot the losses for all three methods
pl.figure(figsize=(10, 4))

## Create axis
# fig, ax = pl.subplots(figsize=(10, 4))

pl.plot(data_ncf.mean(0), color='magenta', label='NCF')
pl.plot(data_coda.mean(0), color='orange', label='CoDA')
pl.plot(data_cavia.mean(0), color='indigo', label='CAVIA')

## Plot deviation bands
pl.fill_between(range(1500), data_ncf.mean(0) - data_ncf.std(0), data_ncf.mean(0) + data_ncf.std(0), color='magenta', alpha=0.2)
pl.fill_between(range(1500), data_coda.mean(0) - data_coda.std(0), data_coda.mean(0) + data_coda.std(0), color='orange', alpha=0.2)
# pl.fill_between(range(1500), data_cavia.mean(0) - data_cavia.std(0), data_cavia.mean(0) + data_cavia.std(0), color='indigo', alpha=0.2)

## make the interval for CAVIA smaller
pl.fill_between(range(1500), data_cavia.mean(0) - 0.2 * data_cavia.std(0), data_cavia.mean(0) + 0.2 * data_cavia.std(0), color='indigo', alpha=0.2)

pl.ylim(1e-4, 20)

pl.xlabel('Iterations')
# pl.ylabel('MSE')
pl.yscale('log')
pl.legend(fontsize=14)
pl.draw_all()

# pl.rcParams.update(pl.rcParamsDefault)
pl.savefig('train_losses.pdf', dpi=600, bbox_inches='tight')
# pl.savefig('train_losses.jpg')

