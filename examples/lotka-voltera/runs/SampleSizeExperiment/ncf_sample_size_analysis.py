#%%
import pandas as pd

## Assign each folder a new name
ncf_sample_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]
folder_names = ['16022024-115227', '16022024-122638', '16022024-131350', '16022024-140720', '16022024-150923', '16022024-161301', '16022024-172037', '16022024-183601', '16022024-195713']


data1 = [1, 1.01e-04, 7.95e-06, 2.51e-04, 1.88e-04]

data2 = [2, 7.45e-05, 7.89e-06, 1.74e-04, 9.28e-05]

data3 = [3, 7.65e-05, 7.14e-06, 1.36e-04, 5.14e-05]

data4 = [4, 8.41e-05, 7.27e-06, 1.78e-04, 1.01e-04]

data5 = [5, 9.05e-05, 2.65e-05, 1.74e-04, 8.63e-05]

data6 = [6, 5.73e-05, 5.60e-06, 7.52e-05, 3.47e-05]

data7 = [7, 5.86e-05, 6.36e-06, 9.97e-05, 3.66e-05]

data8 = [8, 5.91e-05, 3.82e-06, 1.27e-04, 3.74e-05]

data9 = [9, 6.59e-05, 1.86e-05, 1.11e-04, 2.36e-05]

## Create a dataframe with each line a data point
df_1 = pd.DataFrame(data = [data1, data2, data3, data4, data5, data6, data7, data8, data9], columns = ['ncf_sample_size', 'mean_ind', 'std_ind', 'mean_ood', 'std_ood'])

## Plot boxes at means, betweeen mean-stds and mean+stds of the in-domain data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 6))

def plot_errorbars(arg, **kws):
    np.random.seed(sum(map(ord, "error_bars")))
    x = np.random.normal(0, 1, 100)
    f, axs = plt.subplots(2, figsize=(7, 2), sharex=True, layout="tight")
    sns.pointplot(x=x, errorbar=arg, **kws, capsize=.3, ax=axs[0])
    sns.stripplot(x=x, jitter=.3, ax=axs[1])

# plot_errorbars("sd")


# sns.pointplot(df_1, x="ncf_sample_size", y="mean_ind", errorbar="sd")

# tips = sns.load_dataset("tips")
# tips



#%%

folder_names = ['16022024-115227', '16022024-122638', '16022024-131350', '16022024-140720', '16022024-150923', '16022024-161301', '16022024-172037', '16022024-183601', '16022024-195713']
mean_order = np.argsort(df_1["mean_ind"].to_numpy())
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange']
# ## colors such that the brighter the color, the higher the mean
# colors = sns.color_palette("magma", len(folder_names))
# colors = np.array(colors)[mean_order]


## Load the tests scores from each folder's anaylsis subfolder
for i, folder in enumerate(folder_names):
    print(i+1, folder)
    ## Load the test scores
    scores = pd.read_csv(f'runs/{folder}/analysis/test_scores.csv')
    # print(scores)
    ## Load the ncf sample size
    ncf_sample_size = np.array([i+1]*10)
    scores['ncf_sample_size'] = ncf_sample_size
    scores['color'] = colors[i]*10
    if i==0:
        df = scores
    else:
        df = pd.concat([df, scores])
print(df.head(10))



sns.set_theme(context='talk', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 3})
# sns.set_theme(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 6))
ax.set(yscale="log")
# ax.legend(False)

# sns.catplot(df, x="ncf_sample_size", y="ind_crit", errorbar="se", label="In-Domain", kind="bar")

# sns.pointplot(df, x="ncf_sample_size", y="ind_crit", errorbar="se", hue="color", ax=ax, legend=False, dodge=True, log_scale=True, palette="magma")
#make one plot for the line without points and errorbars
sns.pointplot(df, x="ncf_sample_size", y="ind_crit", markers="", ci=None, color="k", alpha=0.2)
sns.pointplot(df, x="ncf_sample_size", y="ind_crit", errorbar="se", ax=ax, legend=False, log_scale=True, hue="color")

# sns.pointplot(df, x="ncf_sample_size", y="ood_crit", errorbar="se", label="OOD")

ax.set(xlabel='Neighbour Sample Size', ylabel='Log MSE')
# ax.grid(True)