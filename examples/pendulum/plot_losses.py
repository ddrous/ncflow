#%%
from nodax import *
## Set the line width of the plots to 4
plt.rcParams['lines.linewidth'] = 4

ofa_folder = "runs/18112024-160214-OneForAll"

losses_ofa = []
val_losses_ofa = []
for index in range(1,3+1):
    train_loss = np.load(ofa_folder + f"-{index}/train_histories.npz")['losses_node'][:2000].squeeze()
    print(train_loss.shape)
    losses_ofa.append(train_loss)
    val_loss = np.load(ofa_folder + f"-{index}/val_losses.npy").squeeze()[:6,1]
    val_losses_ofa.append(val_loss)

## Doensample to 1000, then plot the mean while shading the std
losses_ofa = [loss[::1] for loss in losses_ofa]
losses_ofa = np.array(losses_ofa)

print(losses_ofa.shape)
mean_ofa = np.mean(losses_ofa, axis=0)
std_ofa = np.std(losses_ofa, axis=0)

val_losses_ofa = np.stack(val_losses_ofa, axis=0)
mean_val_ofa = np.mean(val_losses_ofa, axis=0)

print(mean_ofa)

#%%

## Now let's collect the OnePerEnv Losses
ope_folder = "runs/18112024-123710-OnePerEnv-OoD"
ope_indices = [0, 1]
losses_ope = []
val_losses_ope = []
for index in ope_indices:
    train_loss = np.load(ope_folder + f"/adapt_{index}/train_histories.npz")['losses_node'].squeeze()
    print(train_loss.shape)
    losses_ope.append(train_loss)
    val_loss = np.load(ope_folder + f"/adapt_{index}/val_losses.npy").squeeze()[...,1]
    val_losses_ope.append(val_loss)

## Doensample to 2000, then plot the mean while shading the std
losses_ope = [loss[::5] for loss in losses_ope]
losses_ope = np.array(losses_ope)

losses_ope = np.clip(losses_ope, 1e-4, None)
mean_ope = np.mean(losses_ope, axis=0)
std_ope = np.std(losses_ope, axis=0)

val_losses_ope = np.array(val_losses_ope)
print(val_losses_ope.shape)
mean_val_ope = np.mean(val_losses_ope, axis=0)
std_val_ope = np.std(val_losses_ope, axis=0)

# ## Take the exponential moving average of the standard deviation
# new_std_ope = np.zeros_like(std_ope)
# alpha = 0.999
# for i in range(1, len(std_ope)):
#     new_std_ope[i] = alpha*new_std_ope[i-1] + (1-alpha)*std_ope[i]
# std_ope = new_std_ope



#%%

ncf_folders = ["runs/18112024-192758-T1-First", "runs/18112024-180943-T1-Second"]
losses_ncf = []
val_losses_ncf = []
for ncf_folder in ncf_folders:
    train_loss = np.load(ncf_folder + f"/train_histories.npz")['losses_node'].squeeze()
    # print(np.load(ncf_folder + f"/train_histories.npz").files)
    val_loss = np.load(ncf_folder + f"/val_losses.npy").squeeze()[...,1]
    print(val_loss.shape)
    val_losses_ncf.append(val_loss)
    losses_ncf.append(train_loss)

## Doensample to 2000, then plot the mean while shading the std
losses_ncf = [loss[::6] for loss in losses_ncf]
losses_ncf = np.array(losses_ncf)

losses_ncf = np.clip(losses_ncf, 1e-4, None)
mean_ncf = np.mean(losses_ncf, axis=0)
std_ncf = np.std(losses_ncf, axis=0)

mean_val_ncf = np.mean(val_losses_ncf, axis=0)[::6]
std_val_ncf = np.std(val_losses_ncf, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
print(mean_ofa)
progress = np.linspace(0, 1, mean_ofa.shape[0])*100

##=========
ax.plot(progress, mean_ofa, label="OFA", color="blue")
ax.fill_between(progress, mean_ofa-std_ofa, mean_ofa+std_ofa, alpha=0.3, color="blue")

progress_val_ofa = np.linspace(0, 1, mean_val_ofa.shape[0])*100
ax.plot(progress_val_ofa, mean_val_ofa, "x-", color="blue", markersize=10, alpha=0.5, lw=1)

##=========
ax.plot(progress, np.mean(losses_ope, axis=0), label="OPE", color="green")
ax.fill_between(progress, mean_ope-std_ope, mean_ope+std_ope, alpha=0.3, color="green")
progress_val_ope = np.linspace(0, 1, val_losses_ope[0].shape[0])*100
ax.plot(progress_val_ope, mean_val_ope, "x-", color="green", markersize=5, alpha=0.5, lw=1)
## Interpolate mean_val_ope to have 124 poitns 

##=========
ax.plot(progress, np.mean(losses_ncf, axis=0), label="NCF", color="red")
ax.fill_between(progress, mean_ncf-std_ncf, mean_ncf+std_ncf, alpha=0.3, color="red")

progress_val_ncf = np.linspace(0, 1, mean_val_ncf.shape[0])*100
ax.plot(progress_val_ncf, mean_val_ncf, "x-", color="red", markersize=10, alpha=0.5, lw=1)
# ax.fill_between(progress_val, mean_val_ncf-std_val_ncf, mean_val_ncf+std_val_ncf, alpha=0.3, color="purple")

## The X-axis is in Training progress, while the Y-axis is in Loss
ax.set_xlabel("Training Progress (%)", fontsize=16)
# ax.set_ylabel("MSE", fontsize=16)
ax.set_yscale("log")

ax.legend(fontsize=20, loc='lower left')
plt.show()


## Save the figure
fig.savefig("tmp/ofa_ope_ncf.pdf", dpi=300, bbox_inches='tight')