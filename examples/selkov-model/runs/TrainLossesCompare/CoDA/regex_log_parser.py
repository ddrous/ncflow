#%%
import re
import matplotlib.pyplot as plt
import numpy as np

# get the content of the nohup.log file as a string
with open('nohup.log', 'r') as file:
    code = file.read()

# ## Or read from the log file in forced/20240304_194521
# with open('results/forced/20240304_194521/log', 'r') as file:
#     code = file.read()


## collect all the loss values and epochs
loss_train_values = re.findall(r'Loss Train: (\d+\.\d+e[+-]\d+)', code)
loss_train_values = [float(i) for i in loss_train_values]
epochs_train = re.findall(r'Epoch (\d+), Iter \d+, Loss Train:', code)
epochs_train = [int(i) for i in epochs_train]

## collect all the loss values and epochs
loss_test_values = re.findall(r'Loss Test ind: (\d+\.\d+e[+-]\d+)', code)
# loss_test_values = re.findall(r'Loss Test: (\d+\.\d+e[+-]\d+)', code)
loss_test_values = [float(i) for i in loss_test_values]
epochs_test = re.findall(r'Epoch (\d+), Iter \d+, Loss Test ind:', code)
# epochs_test = re.findall(r'Epoch (\d+), Iter \d+, Loss Test:', code)
epochs_test = [int(i) for i in epochs_test]

## plot the loss values

## Set figure size
plt.figure(figsize=(10, 5))

plt.plot(epochs_train, loss_train_values, label='Loss Train')
plt.plot(epochs_test, loss_test_values, label='Loss Test')

plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.yscale('log')
plt.ylim(1e-5, 0.1)
plt.legend()
plt.draw()

#%%

## Final loss values
print('Final Loss Train: ', loss_train_values[-1])
print('Final Loss Test: ', loss_test_values[-1])

# %%

## save the figure to selkov.pdf
# plt.savefig('selkov.pdf', dpi=300, bbox_inches='tight')