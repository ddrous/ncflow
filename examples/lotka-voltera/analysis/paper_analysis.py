#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Load data from test_score.csv
df = pd.read_csv('test_scores_2.csv', sep=',', header=0)
df

# df[""]

## Compute the mean accross the ind colum
# stop = 10
mean_ind = df.loc[:, 'ind'].mean()
print(mean_ind)

std_ind = df.loc[:, 'ind'].std()
print(std_ind)

# df.boxplot(column=['ind'], grid=False, figsize=(12,8), fontsize=12)


df.describe()



mean_ood = df.loc[:, 'ood'].mean()
print(mean_ood)

std_ood = df.loc[:, 'ood'].std()
print(std_ood)







# seed,ind,ood
# 4000,8.428190994891338e-06,4.5910223889222834e-06
# 4100,1.9342714949743822e-05,5.854154096596176e-06
# 4200,1.2752635484503116e-05,3.490719973342493e-05
# 4300,9.776535989658441e-06,2.031799522228539e-05
# 4400,1.1605340660025831e-05,1.8224045561510138e-06
# 4500,1.6571953892707825e-05,5.271294412523275e-06
# 4600,1.5558392988168634e-05,2.881842192437034e-06
# 4700,1.1904299753950909e-05,1.3434970469461405e-06
# 4800,1.1397885828046128e-05,9.104111995839048e-06
# 4900,1.1745619303837884e-05,1.743886627991742e-06
# 5000,1.7134554582298733e-05,1.4423396351048723e-05
# 5100,1.110219454858452e-05,2.264902832394e-05
# 5200,1.335233264398994e-05,2.91568971988454e-06
# 5300,1.698367668723222e-05,0.00013245444279164076
# 5400,1.3600213605968747e-05,4.289528988010716e-06
# 5500,1.2844424418290146e-05,2.7396235964260995e-06
# 5600,1.0310698598914314e-05,2.328607479284983e-06
# 5700,1.2630783203348983e-05,3.0352341582329245e-06
# 5800,1.3580125596490689e-05,4.271199941285886e-06
# 5900,1.3703637705475558e-05,6.039120853529312e-05




#%%





mapes = np.load("mapes.npy")
print(mapes)
mapes_mean = mapes[mapes<26].mean()
mapes = np.where(mapes<26, mapes, mapes_mean)

envs = [ (b, d) for b in np.linspace(0.25, 1.25, 11) for d in np.linspace(0.25, 1.25, 11)]

envs_arr = [np.array(e) for e in envs]
envs_final = np.vstack(envs_arr)

print(envs_final.shape)
print(mapes.shape)

### Plot trisurf of mapes agains envs_final
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')


# surf = ax.plot_trisurf(envs_final[:,0], envs_final[:,1], mapes, cmap='viridis')
# plt.pcolormesh(envs_final[:,0], envs_final[:,1], mapes, cmap='viridis')

x = envs_final[:,0].reshape((11, 11))
y = envs_final[:,1].reshape((11, 11))
z = mapes.reshape((11, 11)) 
# surf = ax.imshow(z, cmap='viridis')
surf = plt.contourf(x, y, z, cmap='coolwarm', levels=50)


# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set labels and title
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'$\delta$')
ax.set_title('MAPE')

plt.show()





#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(context='talk', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 4})


loss_bad1 = np.load("train_histories_bad.npz")["losses_node"]
loss_bad2 = np.load("train_histories_bad_2.npz")["losses_node"]
loss_bad = np.stack([loss_bad1, loss_bad2], axis=0)
mean_bad = np.mean(loss_bad, axis=0).squeeze()
std_bad = np.std(loss_bad, axis=0).squeeze()

loss_good1 = np.load("train_histories.npz")["losses_node"]
loss_good2 = np.load("train_histories_2.npz")["losses_node"]
loss_good = np.stack([loss_good1, loss_good2], axis=0)
mean_good = np.mean(loss_good, axis=0).squeeze()
std_good = np.std(loss_good, axis=0).squeeze()


num_epochs = 600000

## Plot these two in the same plot nicely
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.plot(mean_bad, label="NCF Without 3-Networks Architecture", color="blue")
# sns.lineplot(x=range(num_epochs), y=mean_bad, label='One-For-All', color='blue', ax=ax)
ax.plot(mean_good, label="NCF Complete", color="crimson")

# print(mean_bad.shape)
ax.fill_between(range(num_epochs), mean_bad - std_bad, mean_bad + std_bad, color='blue', alpha=0.3)
ax.fill_between(range(num_epochs), mean_good - std_good, mean_good + std_good, color='crimson', alpha=0.3)


plt.yscale("log")
plt.xlabel("Epoch", fontsize=24)
plt.ylabel("MSE", fontsize=24)
plt.legend()


plt.savefig("losses_ablation.pdf", bbox_inches='tight', dpi=300)




#%%


## Load the test_scores.csv
df = pd.read_csv('test_scores_3.csv', sep=',', header=0)
df.mean()

## print with better precision
# print(df.mean().to_string(float_format="{:0.2e}".format))
np.std(df.values, axis=0)