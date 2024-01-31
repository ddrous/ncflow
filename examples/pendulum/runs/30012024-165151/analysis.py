#%%

import numpy as np
import pandas as pd

## Load one for all losses in oneforall/
df = pd.read_csv('./analysis/test_scores.csv')
# df.mean(axis=0)
df.std(axis=0)



## New score for SP
# IND: 0.008389 +- 0.000901
# OOD: 1.239647 +- 0.867611


#%%

import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='paper', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 4})
# plt.style.use("dark_background")
# sns.set_style("whitegrid", {'axes.grid' : True})

## Increaset x and y ticks and lable sizes
# plt.rcParams['xtick.labelsize']=14
# plt.rcParams['ytick.labelsize']=14
plt.rcParams['axes.labelsize']=14
plt.rcParams['axes.titlesize']=14

#%%
# Generate some sample data (replace with your actual data)
num_epochs = 6000
num_curves = 5

losses = [np.load(f'analysis/loss_{i}.npy') for i in range(1,3)]
loss_data = np.stack(losses, axis=0)
mean_loss = np.mean(loss_data, axis=0)
std_loss = np.std(loss_data, axis=0)


loss1 = np.load('analysis/train_histories.npz')["losses_node"]
loss2 = np.load('analysis/train_histories_2.npz')["losses_node"]
losses_ = np.stack([loss1, loss2], axis=0)
mean_loss_ = np.mean(losses_, axis=0).squeeze()
std_loss_ = np.std(losses_, axis=0).squeeze()


# Create a figure and an axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the first line with shaded area
sns.lineplot(x=range(num_epochs), y=mean_loss, label='One-For-All', color='blue', ax=ax)
ax.fill_between(range(num_epochs), mean_loss - std_loss, mean_loss + std_loss, color='blue', alpha=0.3)

# Plot the second line with shaded area
sns.lineplot(x=range(num_epochs), y=mean_loss_, label='Neural Context Flow', color='red', ax=ax)
ax.fill_between(range(num_epochs), mean_loss_ - std_loss_, mean_loss_ + std_loss_, color='red', alpha=0.3)

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Set labels for x and y axes
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')

# Set the title of the plot
# ax.set_title('Mean Loss Curve with Standard Deviation Shading')

# Display the legend
ax.legend()

# Show the plot
plt.show()

# Save the figure
fig.savefig('analysis/losses.pdf', transparent=False, dpi=600)




#%%

## Load the losses_node from the train histories in analysis/
loss1 = np.load('analysis/train_histories.npz')["losses_node"]
loss2 = np.load('analysis/train_histories_2.npz')["losses_node"]

# plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
# plt.plot(loss1, label='Loss 1', color='blue')
plt.plot(loss2, label='Loss 2', color='red')










#%%


import equinox as eqx
from nodax import ContextParams

## Load the contexts.eqx

nb_envs = 25
context_size = 1024

skeleton = ContextParams(nb_envs, context_size)
contexts = eqx.tree_deserialise_leaves("contexts.eqx", skeleton).params


skeleton_adapt = ContextParams(2, context_size)
contexts_adapt = eqx.tree_deserialise_leaves("adapt/adapted_contexts_170846_.pkl", skeleton_adapt).params



# colors = ['dodgerblue', 'r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple', 'brown']
colors = ['dodgerblue', 'b', 'darkblue', 'skyblue', 'turquoise']
colors = colors*(nb_envs)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)

ax.scatter(contexts[:,0], contexts[:,1], s=100, c=colors[:nb_envs], marker='o')
for i, (x, y) in enumerate(contexts[:, :2]):
    ax.annotate(str(i), (x, y), xytext=(x+1e-4, y+1e-3), fontsize=9, weight='bold')

ax.scatter(contexts_adapt[:,0], contexts_adapt[:,1], s=150, c=['crimson', 'crimson'], marker='X')
labels = [r"$e_0$", r"$e_1$"]
for i, (x, y) in enumerate(contexts_adapt[:, :2]):
    # ax.annotate(str(i)+"'", (x, y), xytext=(x+1e-4, y+1e-3), fontsize=9)
    ax.annotate(labels[i], (x, y), xytext=(x-5e-3, y+1e-3), fontsize=12, weight='bold')


ax.set_xlabel(r'dim 0')
ax.set_ylabel(r'dim 1')


# ax.set_title(r'First 2 dimensions of the contexts')

## Save this as transparent high def pdf
# plt.savefig('analysis/representation.png', transparent=True, dpi=300)


## Save this as a pdf
plt.savefig('analysis/representation.pdf', transparent=False, dpi=300)







#%%

## NCF

## Seeds are 1082, 1084, and 1088
# IND: 0.00485, 0.0047, 0.0052
# OOD: 0.00024, 0.00023, 0.0013

ind = [0.00485, 0.0047, 0.0049]
ood = [0.00024, 0.00023, 0.00028]

ind_mean = np.mean(ind)
ind_std = np.std(ind)

ood_mean = np.mean(ood)
ood_std = np.std(ood)

print(f"IND: {ind_mean:.5f} +- {ind_std:.5f}")
print(f"OOD: {ood_mean:.5f} +- {ood_std:.5f}")


#%%

## OFA

ind = [1.1418129, 1.1449171]
ood = [3.2791748, 3.332]

ind_mean = np.mean(ind)
ind_std = np.std(ind)

ood_mean = np.mean(ood)
ood_std = np.std(ood)

print(f"IND: {ind_mean:.5f} +- {ind_std:.5f}")
print(f"OOD: {ood_mean:.5f} +- {ood_std:.5f}")









#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

c = 2
ind1 = [0.2972967, 0.5362495, 0.46570486]
ood1 = [0.12615547, 2.37, 0.9652765]

c= 4
ind2 = [0.13577554, 0.44245037, 0.36009705]
ood2 = [0.23241526, 2.0955732, 0.4132346]


c=16
ind3 = [0.0033225645, 0.011012313, 0.0071533322]
ood3 = [1.8910992, 2.9741187, 2.431864]


c = 512
ind4 = [0.0025011243, 0.008244528, 0.0055813324]
ood4 = [1.8484825, 2.5651355, 2.388569]



## Cloalesce everything
cs = [2, 4, 16, 512]
inds = [ind1, ind2, ind3, ind4]

## We want to plot c angainst the mean and std of ind (using bloxplots)
data = np.stack(inds, axis=0)
# data

mean_data = np.mean(data, axis=1)
std_data = np.std(data, axis=1)

mean_data

## Now do the boxplots
# df = pd.DataFrame(data)
# df = pd.DataFrame(np.log10(data.T), columns=cs)
df = pd.DataFrame(data.T, columns=cs)
print(df)
colors = {'2': 'blue', '4': 'green', '16': 'orange'}
# colors = ['dodgerblue', 'b', 'darkblue', 'skyblue', 'turquoise']
# df.boxplot(vert=True)
# df.violinplot()
# sns.violinplot(data=df, palette="Set3")


# df = pd.DataFrame(data.T, columns=cs)
sns.boxplot(data=df, palette="Set3")
plt.yscale('log')  # Set log scale for y-axis
# plt.grid(True)
# plt.title('Boxplot of Values by Category (Log Scale)')
plt.xlabel('Context Size'+r" $d_{\xi}$", fontsize=14)
plt.ylabel('Log MSE', fontsize=14)
plt.draw()

plt.savefig('analysis/boxplot.pdf', transparent=False, dpi=600, bbox_inches='tight', pad_inches=0.1)






#%%


def plot_traj(traj, model_y, ts, ys, stop=1000):

    # import numpy as np
    # import seaborn as sns
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='talk', style='ticks',
            font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 6})
    # plt.style.use("dark_background")
    # sns.set_style("whitegrid", {'axes.grid' : True})

    ## Increaset x and y ticks and lable sizes
    # plt.rcParams['xtick.labelsize']=14
    # plt.rcParams['ytick.labelsize']=14
    plt.rcParams['axes.labelsize']=14
    plt.rcParams['axes.titlesize']=14

    # fig, ax = plt.subplots(2, 2, figsize=(6*2, 3.5*2))
    fig, ax = plt.subplot_mosaic('A', figsize=(6*2, 3.5*3))
    # model_y, _ = model(ts, ys[traj, 0])   ## TODO predicting on the entire trajectory ==forecasting !
    # model_y = yhat

    ax['A'].plot(ts[:stop], ys[traj, :stop, 0], c="dodgerblue", label="GT Trajectory", lw=4)
    ax['A'].plot(ts[:stop], model_y[:stop], "o-", markersize=8, c="crimson", label="Prediction")

    # ax['A'].plot(ts[:stop], ys[traj, :stop, 1], c="violet", label="Predators (GT)")
    # ax['A'].plot(ts[:stop], model_y[:stop, 1], ".-", c="purple", label="Predators (NODE)")
    
    mse = np.mean((ys[traj, :stop, 0] - model_y[:stop])**2)

    ax['A'].set_xlabel("Time", fontsize=28)
    ax['A'].set_title(f"MSE: {mse:.3f}", fontsize=28)
    ax['A'].legend(fontsize=18, loc="upper right")

    plt.tight_layout()
    plt.savefig("analysis/ncf_traj.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()

raw_data = np.load("train_data.npz")
ys, ts = raw_data['X'], raw_data['t']
ys = np.reshape(ys, (-1, ys.shape[2], ys.shape[3]))

model_y = np.load("analysis/X_hat.npy")[:, 0]


plot_traj(28, model_y, ts, ys)
