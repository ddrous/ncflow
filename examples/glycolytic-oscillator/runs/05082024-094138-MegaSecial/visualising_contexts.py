#%%
from nodax import *

nb_envs = 9
context_size = 256

contexts = ContextParams(nb_envs, context_size, key=None)
contexts = eqx.tree_deserialise_leaves('contexts.eqx', contexts)

print(contexts.params)

X = contexts.params


contexts_adapt = ContextParams(4, context_size, key=None)
contexts_adapt = eqx.tree_deserialise_leaves('adapt/adapted_contexts_170846_.pkl', contexts_adapt)

X = np.concatenate((X, contexts_adapt.params))


#%%

## Do a t-SNE plot of the contexts in 2D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='notebook', style='white')

X_embedded = TSNE(n_components=2, perplexity=2).fit_transform(X)

#%%

# plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
# plt.show()

plt.figure(figsize=(5, 5))
## Plot the training context differetn from the adapted contexts
plt.scatter(X_embedded[:9, 0], X_embedded[:9, 1], color="green", s=200, label='Training')
plt.scatter(X_embedded[9:, 0], X_embedded[9:, 1], marker="X", color="purple", s=200, label='Adaptation')

## Add anotations as the ids of the contexts for the training (in the center of the point)
for i in range(9):
    # plt.annotate(i, (X_embedded[i, 0], X_embedded[i, 1]), fontsize=10)
    plt.text(X_embedded[i, 0], X_embedded[i, 1], str(i), fontsize=10, ha='center', va='center')

## Add anotations as the ids of the contexts for the adaptation (in the top left of the point)
for i in range(9, 13):
    plt.text(X_embedded[i, 0]+20, X_embedded[i, 1]+20, str(i-9), fontsize=10, ha='left', va='top')

plt.xlabel(r't-SNE $1$')
# plt.ylabel(r't-SNE $2$')
plt.ylabel('t-SNE 2')
# plt.legend()


## Save the plot
plt.savefig('tsne_contexts_go.pdf', dpi=300,bbox_inches='tight')

plt.draw()
