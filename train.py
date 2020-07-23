# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 01:42:43 2020

@author: TOSHIBA
"""

import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("white")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pylab import rcParams
#df=pd.read_csv("arac_tahminfiyatlı.csv")
df=pd.read_csv("tam_arac_tahmin.csv")

df=df.dropna()
df=df.drop("Başlik", axis=1)

df.rename(columns={'Price;': 'Price'}, inplace=True)
df.Yil = pd.to_numeric(df.Yil, errors = 'coerce', downcast= 'integer')

#df=df[df.Yil > 2002]
#%%
df = df[df.Price.str.contains("TL;") == True]
# remove the 'DH' caracters from the price
df.Price = df.Price.map(lambda x: x.rstrip('TL;'))
# remove the space on it
df.Price = df.Price.str.replace(" ","")
# change it to integer value
df.Price = pd.to_numeric(df.Price, errors = 'coerce', downcast= 'integer')
df.Kilometre = pd.to_numeric(df.Kilometre, errors = 'coerce', downcast= 'integer')
df.Yil = pd.to_numeric(df.Yil, errors = 'coerce', downcast= 'integer')

#%%
df=df.dropna()

data=df
X = data[['Marka', 'Vites', 'Yakit','Kilometre','Yil','Seri']]
Y = data.Price
X = pd.get_dummies(data=X)
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
#%%              TRAIN

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6, learning_rate=0.08,n_estimators=1000)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)


print('Variance score: %.2f' % r2_score(Y_test, predicted))

rmse = np.sqrt(mean_squared_error(Y_test, predicted))
scores = cross_val_score(gbr, X, Y, cv=11)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)

#%%   SAVE THE BEST MODEL
from sklearn.externals import joblib

joblib.dump(gbr, 'model.pkl')
gbr = joblib.load('model.pkl')



