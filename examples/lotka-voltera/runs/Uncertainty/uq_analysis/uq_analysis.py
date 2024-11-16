#%%
from nodax import *

from matplotlib.ticker import ScalarFormatter, FuncFormatter


#%%

problems = ["lv", "go", "sm", "bt", "gs", "ns"]
time_steps = [20, 20, 11, 20, 10, 10]
desired_plot_steps = [t + (t//2)+1 for t in time_steps]

stds_ind = []
stds_ood = []

for problem, final_step in zip(problems, desired_plot_steps):
    std_ind = jnp.load(f"data/uq_metrics_{problem}_ind.npz")['rel_std']
    std_ood = jnp.load(f"data/uq_metrics_{problem}_ood.npz")['rel_std']
    stds_ind.append(std_ind[:final_step])
    stds_ood.append(std_ood[:final_step])

print([std.shape for std in stds_ind])

normalised_times = [jnp.linspace(0, 1.5, std.shape[0]) for std in stds_ind]

fig, ax = plt.subplots(2, 3, figsize=(6*3, 3*2), sharex=True, sharey=False)
ax = ax.flatten()
for i, (normalised_time, std_ind, std_ood) in enumerate(zip(normalised_times, stds_ind, stds_ood)):
    ax[i].plot(normalised_time, std_ind, label="In-Domain", color='royalblue', lw=4)
    ax[i].plot(normalised_time, std_ood, label="Out-of-Distribution", color='crimson', lw=4)
    ax[i].axvline(1, color='black', linestyle='--', label="Forecast Start", alpha=0.5)
    ax[i].set_title(problems[i].upper())
    if i > 2:
        ax[i].set_xlabel("Normalized Time", fontsize=16)
    if i % 3 == 0:
        if i == 0:
            ax[i].set_ylabel(f"Std. Dev. $\hat \sigma$", fontsize=16, labelpad=15)
        else:
            ax[i].set_ylabel(f"Std. Dev. $\hat \sigma$", fontsize=16)
    if i == 0:
        ax[i].legend()
    ## Values on the y axis should be in scientific notation
    ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # Bold the title
    ax[i].title.set_fontweight('bold')
    ax[i].title.set_fontsize(16)

plt.tight_layout()
plt.savefig("std_devs.pdf", dpi=300, bbox_inches='tight')


# %%
## Plot the standsrd deviations vs errors

# problems[-1] = "ns_"

fig, ax = plt.subplots(2, 3, figsize=(6*3, 4*2), sharex=False, sharey=False)
ax = ax.flatten()
for i, problem in enumerate(problems):

    deviations_ind = jnp.load(f"data/uq_metrics_{problem}_ind.npz")['deviations']
    deviations_ood = jnp.load(f"data/uq_metrics_{problem}_ood.npz")['deviations']

    errors_ind = jnp.load(f"data/uq_metrics_{problem}_ind.npz")['errors']
    errors_ood = jnp.load(f"data/uq_metrics_{problem}_ood.npz")['errors']

    ax[i].scatter(deviations_ind, errors_ind, label="In-Domain", color='royalblue')
    ax[i].scatter(deviations_ood, errors_ood, label="Out-of-Distribution", color='crimson', alpha=0.4, marker='x')

    ax[i].set_title(problems[i].upper())
    if i % 3==0:
        if i != 0:
            ax[i].set_ylabel("Error $|x - \hat \mu |$", fontsize=16, labelpad=15)
        else:
            ax[i].set_ylabel("Error $|x - \hat \mu |$", fontsize=16)
    if i>2:
        ax[i].set_xlabel(f"Std. Dev. $\hat \sigma$", fontsize=16)
    if i == 0:
        ax[i].legend()

    ## Values on the y axis should be in scientific notation
    ax[i].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[i].ticklabel_format(axis='x', style='sci', scilimits=(0,0))

    # Bold the title
    ax[i].title.set_fontweight('bold')
    ax[i].title.set_fontsize(16)

plt.tight_layout()

plt.savefig("std_devs_vs_errors.png", dpi=100, bbox_inches='tight')