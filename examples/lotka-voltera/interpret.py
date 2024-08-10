#%%

""" Fits the contexts to the underlying paramters """

import equinox as eqx
from nodax import ContextParams
import numpy as np
from sklearn.linear_model import LinearRegression

run_folder = "runs/10082024-101628/"


#%%

nb_envs, context_size = 9, 2

# Load the contxts with equinox
contexts = ContextParams(nb_envs, context_size, key=None)
X_train = eqx.tree_deserialise_leaves(run_folder+"contexts.eqx", contexts).params
X_train


# Get the underlying paramters (outputs)
environments = [
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
Y_train = [[env["beta"], env["delta"]] for env in environments]
Y_train = np.array(Y_train)
Y_train

## Fit a OLS model to find a matrix from X_train to Y_train
reg = LinearRegression(fit_intercept=True).fit(X_train, Y_train)

# Get the learned matrix
reg.coef_
print(f"Learned matrix\n {reg.coef_}")



#%%


nb_envs, context_size = 4, 2

# Load the contxts with equinox
contexts = ContextParams(nb_envs, context_size, key=None)
X_test = eqx.tree_deserialise_leaves(run_folder+"adapt/adapted_contexts_170846_.pkl", contexts).params
X_test

environments = [
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 0.625, "gamma": 0.5, "delta": 1.125},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 0.625},
    {"alpha": 0.5, "beta": 1.125, "gamma": 0.5, "delta": 1.125},
]
Y_test = [[env["beta"], env["delta"]] for env in environments]
Y_test = np.array(Y_test)
Y_test


## Evaluate the learned regression model
Y_pred = reg.predict(X_test)
Y_pred

## Compute the error
error = np.mean((Y_pred - Y_test)**2)

## Print the error in scientific notation
print(f"Mean Squared Error: {error:.2e}")



#%%
""" Plot Y_pred vs Y_test, same for trainning """
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='whitegrid', palette='colorblind')

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

Y_train_pred = reg.predict(X_train)
train_color = "orange"
val_color = "orange"
test_color = "purple"

ax.scatter(Y_train[:, 0], Y_train[:, 1], label='Train GT', color=train_color, alpha=0.5)
ax.plot(Y_train_pred[:, 0], Y_train_pred[:, 1], "X", markersize=10, label='Train Pred', color=val_color)

ax.scatter(Y_test[:, 0], Y_test[:, 1], label='Adapt GT', color=test_color, alpha=0.5)
ax.plot(Y_pred[:, 0], Y_pred[:, 1], "X", markersize=10, label='Adapt Pred', color='purple')


ax.set_xlim([0.25, 1.25])
ax.set_ylim([0.25, 1.25])
ax.set_xlabel(r'$\beta$', fontsize=12)
ax.set_ylabel(r'$\delta$', fontsize=12)

## Place xticks in increments of 0.25
xticks = np.arange(0.25, 1.26, 0.25)
yticks = np.arange(0.25, 1.26, 0.25)
ax.set_xticks(xticks)
ax.set_yticks(yticks)
ax.grid(True, which='both', linestyle='--', linewidth=0.3)


ax.legend(loc='lower left', fontsize=8)

## We have nine points in total for trainng. Plot lines connecting the 9 GT points to each other. The lines should form a grid, with 0.75, 0.75 as the center
train_mkln = "--"
for i in range(3):
    # ax.plot([Y_train[i, 0], Y_train[i+3, 0]], [Y_train[i, 1], Y_train[i+3, 1]], train_mkln, color=train_color, alpha=0.5)
    ax.plot([Y_train[i, 0], Y_train[i+6, 0]], [Y_train[i, 1], Y_train[i+6, 1]], color=train_color, alpha=0.5)
    # ax.plot([Y_train[i+3, 0], Y_train[i+6, 0]], [Y_train[i+3, 1], Y_train[i+6, 1]], color=train_color, alpha=0.5)


print("Y_train: \n", Y_train)
## Good. Now plot horizontal lines as well as the vertical lines 
for i in range(1):
    ax.plot([Y_train[i, 0], Y_train[i+2, 0]], [Y_train[i, 1], Y_train[i+2, 1]], color=train_color, alpha=0.5)
    ax.plot([Y_train[i+3, 0], Y_train[i+5, 0]], [Y_train[i+3, 1], Y_train[i+5, 1]], color=train_color, alpha=0.5)
    ax.plot([Y_train[i+6, 0], Y_train[i+8, 0]], [Y_train[i+6, 1], Y_train[i+8, 1]], color=train_color, alpha=0.5)

# ## Repeat the same for the 4 available test points
for i in range(1):
    ax.plot([Y_test[i, 0], Y_test[i+1, 0]], [Y_test[i, 1], Y_test[i+1, 1]], color=test_color, alpha=0.5)
    ax.plot([Y_test[i+2, 0], Y_test[i+3, 0]], [Y_test[i+2, 1], Y_test[i+3, 1]], color=test_color, alpha=0.5)

## Good. Now plot horizontal lines as well as the vertical lines
for i in range(1):
    ax.plot([Y_test[i, 0], Y_test[i+2, 0]], [Y_test[i, 1], Y_test[i+2, 1]], color=test_color, alpha=0.5)
    ax.plot([Y_test[i+1, 0], Y_test[i+3, 0]], [Y_test[i+1, 1], Y_test[i+3, 1]], color=test_color, alpha=0.5)


## Add numbers 0,to 9 next to each training point, as well the train pred points. Place the text sligtyly above, to the upper left of the points
for i in range(9):
    ax.text(Y_train[i, 0], Y_train[i, 1], f"{i}", fontsize=8, color="grey", ha='right', va='bottom')
    ax.text(Y_train_pred[i, 0], Y_train_pred[i, 1], f"{i}", fontsize=6, color="k", ha='left', va='bottom', fontstyle='italic', fontweight='bold')

## Do the same for the test points
for i in range(4):
    ax.text(Y_test[i, 0]-0.01, Y_test[i, 1], f"{i}", fontsize=8, color="grey", ha='right', va='bottom')
    ax.text(Y_pred[i, 0]+0.01, Y_pred[i, 1]+0.01, f"{i}", fontsize=6, color="k", ha='left', va='bottom', fontstyle='italic', fontweight='bold')

## Set the title as CoDA - Error: 1.23e-4
ax.set_title(f"Adapt MSE: {error:.2e}")

## Save the figure
fig.savefig(run_folder+"interpretable_NCF.pdf", dpi=300, bbox_inches='tight')