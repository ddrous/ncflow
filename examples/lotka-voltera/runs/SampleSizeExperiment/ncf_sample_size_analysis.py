#%%
import pandas as pd

## Assign each folder a new name
ncf_sample_sizes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
folder_names = ['17022024-105655', '16022024-115227', '16022024-122638', '16022024-131350', '16022024-140720', '16022024-150923', '16022024-161301', '16022024-172037', '16022024-183601', '16022024-195713']


data0 = [0, 1.26e-04, 1.51e-05, 1.15e-03, 3.17e-04]

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
df_1 = pd.DataFrame(data = [data0, data1, data2, data3, data4, data5, data6, data7, data8, data9], columns = ['ncf_sample_size', 'mean_ind', 'std_ind', 'mean_ood', 'std_ood'])

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

mean_order = np.argsort(df_1["mean_ind"].to_numpy())
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
# ## colors such that the brighter the color, the higher the mean
# colors = sns.color_palette("magma", len(folder_names))
# colors = np.array(colors)[mean_order]


## Load the tests scores from each folder's anaylsis subfolder
for i, folder in enumerate(folder_names):
    print(i, folder)
    ## Load the test scores
    scores = pd.read_csv(f'{folder}/analysis/test_scores.csv')
    # print(scores)
    ## Load the ncf sample size
    ncf_sample_size = np.array([i]*10)
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
ax.set_xticklabels([str(0)+"*", 1, 2, 3, 4, 5, 6, 7, 8, 9])

## Save figure as a pdf
plt.savefig('ncf_sample_size_analysis_ind.pdf', format='pdf', dpi=1200, bbox_inches='tight')

#%%

f, ax = plt.subplots(figsize=(10, 6))
ax.set(yscale="log")
sns.pointplot(df, x="ncf_sample_size", y="ood_crit", markers="", ci=None, color="k", alpha=0.2)
sns.pointplot(df, x="ncf_sample_size", y="ood_crit", errorbar="se", ax=ax, legend=False, log_scale=True, hue="color")
ax.set(xlabel='Neighbour Sample Size', ylabel='Log MSE')
ax.set_xticklabels([str(0)+"*", 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.savefig('ncf_sample_size_analysis_ood.pdf', format='pdf', dpi=1200, bbox_inches='tight')

#%%

## Training times
training_times = [[0, 0, 21, 25]+data0[1:3],
                  [1, 0, 24, 27]+data1[1:3],
                  [2, 0, 35, 46]+data2[1:3],
                  [3, 0, 40, 41]+data3[1:3],
                  [4, 0, 49, 43]+data4[1:3],
                  [5, 0, 51, 47]+data5[1:3],
                  [6, 0, 56, 3]+data6[1:3],
                  [7, 1, 2, 55]+data7[1:3],
                  [8, 1, 8, 37]+data8[1:3],
                  [9, 1, 14, 8]+data9[1:3]]

adaptation_times = [[0, 0, 8, 8]+data0[3:],
                    [1, 0, 9, 15]+data1[3:],
                    [2, 0, 10, 57]+data2[3:],
                    [3, 0, 12, 20]+data3[3:],
                    [4, 0, 11, 51]+data4[3:],
                    [5, 0, 11, 22]+data5[3:],
                    [6, 0, 11, 3]+data6[3:],
                    [7, 0, 12, 0]+data7[3:],
                    [8, 0, 12, 7]+data8[3:],
                    [9, 0, 12, 17]+data9[3:]]

## Create a dataframe with the ncf sample size, training times and adaptation times
df_2 = pd.DataFrame(data = training_times, columns = ['ncf_sample_size', 'hours', 'minutes', 'seconds', 'mean_ind', 'std_ind'])
df_2['color'] = colors

## Build df_3 with adaptation times
df_3 = pd.DataFrame(data = adaptation_times, columns = ['ncf_sample_size', 'hours', 'minutes', 'seconds', 'mean_ood', 'std_ood'])
df_3['color'] = colors


## Add a colum for total training times in seconds
# df_2['total_training_time'] = df_2['hours']*3600 + df_2['minutes']*60 + df_2['seconds']
df_2['total_training_time'] = df_2['hours']*60 + df_2['minutes'] + df_2['seconds']/60
df_3['total_adaptation_time'] = df_3['hours']*60 + df_3['minutes'] + df_3['seconds']/60

## Plot mean_ind against total_training_time
f, ax = plt.subplots(figsize=(10, 6))
ax.set(yscale="log")
# ax.set(xscale="log")
# sns.pointplot(df_2, x="ncf_sample_size", y="mean_ind", ax=ax, markers="", ci=None, color="y", alpha=0.2)
# sns.pointplot(df_2, x="ncf_sample_size", y="mean_ind", ax=ax , markers="o", hue='color', legend=False)

sns.pointplot(df_2, x="ncf_sample_size", y="mean_ind", ax=ax , markers="o", color="red", label="In-Domain Loss")
sns.pointplot(df_3, x="ncf_sample_size", y="mean_ood", ax=ax , markers="o", color="crimson", label="OOD Loss", alpha=0.2)
ax.set_ylabel('Log MSE')

ax2 = ax.twinx()
# sns.pointplot(df_2, x="ncf_sample_size", y="total_training_time", ax=ax2, markers="", ci=None, color="k", alpha=0.2)
# sns.pointplot(df_2, x="ncf_sample_size", y="total_training_time", ax=ax2, markers="s", hue='color', legend=False)

sns.pointplot(df_2, x="ncf_sample_size", y="total_training_time", ax=ax2, markers="s", color="blue", label="Training Time")
sns.pointplot(df_3, x="ncf_sample_size", y="total_adaptation_time", ax=ax2, markers="s", color="skyblue", label="Adaptation Time", alpha=0.5)
ax2.set_ylabel('Time (mins)')

ax2.set_xticklabels([str(0)+"*", 1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xlabel('Neighbour Sample Size')

ax.legend()
ax2.legend()

## Save figure as a pdf
plt.savefig('ncf_sample_size_analysis_training_time.pdf', format='pdf', dpi=1200, bbox_inches='tight')
