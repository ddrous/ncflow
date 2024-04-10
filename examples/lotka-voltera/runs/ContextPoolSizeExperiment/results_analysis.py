#%%
import pandas as pd

## Assign each folder a new name
ncf_sample_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9]


losses1 = [1, 5.224747e-05, 6.783795e-05, 7.0833776e-05]
losses2 = [2, 5.0034e-05, 6.39028e-05, 7.0070906e-05]
losses3 = [3, 4.799081e-05, 6.0690767e-05, 5.5933084e-05]
losses4 = [4, 4.5672597e-05, 5.9366954e-05, 5.397191e-05]
losses5 = [5, 4.9214803e-05, 6.705255e-05, 6.757589e-05]
losses6 = [6, 4.933634e-05, 6.0242553e-05, 5.828382e-05]
losses7 = [7, 4.6182045e-05, 5.7319776e-05, 6.194386e-05]
losses8 = [8, 4.7197453e-05, 9.5398645e-05, 5.47559e-05]
losses9 = [9, 4.4913377e-05, 5.6144734e-05, 5.2251427e-05]

times1 = [1, 51*60+21, 34, 26, 26, 31, 59]
times2 = [2, (60+13)*60+31, 34, 26, 27, 29, 68]
times3 = [3, (60+24)*60+24, 33, 26, 27, 29, 71]
times4 = [4, (60+35)*60+28, 34, 26, 28, 27, 72]
times5 = [5, (60+45)*60+15, 34, 26, 29, 28, 73]
times6 = [6, (60+57)*60+13, 33, 26, 29, 28, 72]
times7 = [7, (120+10)*60+17, 33, 25, 29, 27, 72]
times8 = [8, (120+22)*60+50, 34, 26, 29, 27, 72]
times9 = [9, (120+32)*60+43, 34, 26, 29, 28, 72]


## Create a dataframe with each line a data point
df_1 = pd.DataFrame(data = [losses1, losses2, losses3, losses4, losses5, losses6, losses7, losses8, losses9], columns = ['pool_size', 'ind', 'ood_seq', 'ood_batch'])

df_2 = pd.DataFrame(data = [times1, times2, times3, times4, times5, times6, times7, times8, times9], columns = ['pool_size', 'ind_train (s)', 'ood_seq_1 (s)', 'ood_seq_2 (s)', 'ood_seq_3 (s)', 'ood_seq_4 (s)', 'ood_batch (s)'])
df_2['train_time'] = df_2['ind_train (s)']/60.
df_2['adapt_time'] = df_2['ood_batch (s)']/60.


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(style="darkgrid")
sns.set_context("poster")
f, (ax) = plt.subplots(figsize=(7, 6))

## Plot the in-domain loss vs the pool size
sns.pointplot(data=df_1, x="pool_size", y="ind", ax=ax, markers="o", color="purple", label="In-Domain")
sns.pointplot(data=df_1, x="pool_size", y="ood_batch", ax=ax, markers="o", color="green", label="OOD")

ax.legend()
ax.set(xlabel='Context Pool Size', ylabel='MSE')


## Save figure as a pdf
plt.savefig('context_pool_size_losses.pdf', format='pdf', dpi=600, bbox_inches='tight')



#%%








## Plot the train and adapt times
# ax2 = ax.twinx()
f, ax2 = plt.subplots(figsize=(7, 6))
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()

sns.pointplot(data=df_2, x="pool_size", y="train_time", ax=ax2, markers="s", color="blue", label="In-Domain")
sns.pointplot(data=df_2, x="pool_size", y="adapt_time", ax=ax2, markers="s", color="red", label="OOD")

ax2.legend()
ax2.set(xlabel='Context Pool Size', ylabel='Wall Time (min)')


plt.savefig('context_pool_size_training_times.pdf', format='pdf', dpi=1200, bbox_inches='tight')
