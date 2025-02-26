## Analysis of the context size experiment data
import numpy as np

#%%

context_log_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# train_mse = []
ind_mse = [0.00021246144, 9.777795e-05, 0.0001268837, 8.8680885e-05, 9.9128556e-05, 6.9889975e-05, 5.75747e-05, 5.821536e-05, 5.5043656e-05, 5.0771156e-05]
ood_mse = [0.00026362634, 0.0001521809, 0.0001427407, 9.639266e-05, 0.00016311432, 0.000111383386, 7.257318e-05, 7.768692e-05, 7.753377e-05, 7.464331e-05]



## Plot the MSEs
# from nodax import *

# fig, ax = plt.subplots(figsize=(6, 4))
# sbplot(context_log_sizes, ind_mse, label='In-distribution', ax=ax)
# sbplot(context_log_sizes, ood_mse, label='Out-of-distribution', ax=ax)

# %%

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 3})



fig, ax = plt.subplots(figsize=(5.2, 4))
ax.plot(context_log_sizes, ind_mse, "ro-", markersize=10, label='In-Domain')
ax.plot(context_log_sizes, ood_mse, "gs-", markersize=10, label='Out-of-Distribution')

ax.set_xticks(context_log_sizes)
ax.set_xticklabels(2**context_log_sizes)

# ax.set_xscale('log')
ax.set_yscale('log')


ax.set_xlabel(r'Context size $d_{\xi}$')
ax.legend()


## Save with tight layout and high dpi
plt.tight_layout()
plt.savefig('context_size_exp.pdf', dpi=300, bbox_inches='tight')