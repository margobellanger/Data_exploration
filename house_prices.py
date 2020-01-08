import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# training data
df_train = pd.read_csv('data/input/train.csv')

# features
# print(df_train.columns)

# statistics summary of the house price feature
print(df_train['SalePrice'].describe())

sns.distplot(df_train['SalePrice'])
#plt.show()

# Relationship with numerical variables¶
#scatter plot grlivarea/saleprice
# We can see a linear relation
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))



# Relationship with categorical features¶
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)

# plt.show()



# feature selection is more important than feature engineering
