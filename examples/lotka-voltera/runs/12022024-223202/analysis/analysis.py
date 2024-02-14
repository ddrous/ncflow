#%%
import pandas as pd
pd.set_option('display.float_format', '{:.2e}'.format)

## Load the test_scores.csv file
test_scores = pd.read_csv('test_scores.csv')
test_scores


test_scores.describe()



# seed	ind_crit	ood_crit
# count	1.00e+01	1.00e+01	1.00e+01
# mean	4.90e+03	4.64e-05	4.16e-05
# std	6.06e+02	2.41e-06	1.81e-05