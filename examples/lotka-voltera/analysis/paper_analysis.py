#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## Load data from test_score.csv
df = pd.read_csv('test_scores.csv', sep=',', header=0)
df

# df[""]

## Compute the mean accross the ind colum
df_mean = df.loc[400:600, 'ind'].mean()
df_mean