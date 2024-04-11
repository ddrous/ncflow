#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


res = 32
test_length = int(10*1.0)
# t_test = t_eval[:test_length]
nb_plot_timesteps = 10

e=3
traj=0

def vec_to_mats(vec_uv, res=32, nb_mats=2):
    """ Reshapes a vector into a set of 2D matrices """
    UV = np.split(vec_uv, nb_mats)
    return [np.reshape(UV[i], (res, res)) for i in range(nb_mats)]

def mats_to_vec(mats, res):
    """ Flattens a set of 2D matrices into a single vector """
    return np.concatenate([np.reshape(mats[i], res * res) for i in range(len(mats))])




## Load X and X_hat from the .npz file
nb_round = "Round2/"
npzfile = np.load(nb_round+"sample_prediction_ncf.npz")
X_ncf = npzfile['X'].squeeze()
X_hat_ncf = npzfile['X_hat'].squeeze()

npzfile = np.load(nb_round+"sample_prediction_coda.npz")
X_coda = npzfile['X'].squeeze().transpose(0, 2, 1, 3, 4)[e].reshape(-1, 2048)
X_hat_coda = npzfile['X_hat'].squeeze().transpose(0, 2, 1, 3, 4)[e].reshape(-1, 2048)

npzfile = np.load(nb_round+"sample_prediction_cavia.npz")
X_cavia = npzfile['X'].squeeze()
X_hat_cavia = npzfile['X_hat'].squeeze()


# print(np.prod(X_ncf.shape), np.prod(X_coda.shape))
# print(X_coda.shape)

X = X_ncf
X_hat = X_hat_ncf
save_path = "sample_prediction_ncf.svg"

# X = X_coda
# X_hat = X_hat_coda
# save_path = "sample_prediction_coda.svg"

# X = X_cavia
# X_hat = X_hat_cavia
# save_path = "sample_prediction_cavia.svg"


#%%
nb_mats = X_hat.shape[1] // (res*res)
assert nb_mats > 0, f"Not enough dimensions to form a {res}x{res} matrix"
# mats = vec_to_mats(X_hat, res, nb_mats)

if test_length < nb_plot_timesteps:
    print(f"Warning: trajectory visualisation length={test_length} is less than number of plots per row={nb_plot_timesteps}.")
    nb_plot_timesteps = 1
    print(f"Setting the number of plots per row to {nb_plot_timesteps}")
elif test_length%nb_plot_timesteps !=0:
    print(f"Warning: trajectory visualisation length={test_length} is not divisible by number of plots per row={nb_plot_timesteps}.")
    nb_plot_timesteps = int(test_length / (test_length//nb_plot_timesteps))
    print(f"Setting the number of plots per row to {nb_plot_timesteps}")

fig, ax = plt.subplots(nrows=nb_mats*2, ncols=nb_plot_timesteps, figsize=(2*nb_plot_timesteps, 2*nb_mats*2))
for j in range(0, test_length, test_length//nb_plot_timesteps):
    gt_j = vec_to_mats(X[j], res, nb_mats)
    ncf_j = vec_to_mats(X_hat[j], res, nb_mats)
    for i in range(nb_mats):
        ax[2*i, j].imshow(gt_j[i], cmap='gist_ncar', interpolation='bilinear', origin='lower')

        plot_quantity = np.abs(gt_j[i] - ncf_j[i])
        # cmap = "gist_ncar"
        cmap = "magma"
        # vmax = None
        vmax = 1e-1

        ## Print min an max of the plot quantity
        # print(f"Min: {np.min(plot_quantity)}")
        # print(f"Max: {np.max(plot_quantity)}")

        ax[2*i+1, j].imshow(plot_quantity, cmap=cmap, interpolation='bilinear', origin='lower', vmin=0, vmax=vmax)

## Remove the ticks and labels
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])

plt.suptitle(f"2D visualisation results for env={e}, traj={traj}", fontsize=20)

plt.tight_layout()
plt.draw();

plt.savefig(save_path, dpi=600, bbox_inches='tight')


# %%