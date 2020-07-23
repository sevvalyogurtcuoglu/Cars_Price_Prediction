# -*- coding: utf-8 -*-
"""
Created on Sun May 10 05:23:50 2020

@author: TOSHIBA
"""

# -*- coding: utf-8 -*-


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

#%%   DECISION TREE REGRESYON
from sklearn.tree import DecisionTreeRegressor

dtr = DecisionTreeRegressor(max_features='auto')
dtr.fit(X_train, Y_train)
predicted = dtr.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='orange')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(Y_test, predicted))
print('RMSE:')
print(rmse)
from sklearn.metrics import r2_score
print('Variance score: %.2f' % r2_score(Y_test, predicted))




#%%       Hyper parameters tuning for Gradient Boosting Regressor
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

r_sq = []
deep = []
mean_scores = []
lo=[]
#loss =['ls', 'lad', 'huber', 'quantile']
for n in range(3, 11):
    gbr = GradientBoostingRegressor(loss = 'lad', max_depth=n)
    gbr.fit (X, Y)
    deep.append(n)
    r_sq.append(gbr.score(X, Y))
    mean_scores.append(cross_val_score(gbr, X, Y, cv=12).mean())
    #lo.append(gbr.score(X, Y))
plt_gbr = pd.DataFrame()

plt_gbr['mean_scores'] = mean_scores
plt_gbr['depth'] = deep
plt_gbr['R²'] = r_sq

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='R²')
plt.show()

f, ax = plt.subplots(figsize=(15, 5))
sns.barplot(data = plt_gbr, x='depth', y='mean_scores')
plt.show()  """  
#%%
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

gbr = GradientBoostingRegressor(loss ='huber', max_depth=6, learning_rate=0.08,n_estimators=1000)
gbr.fit (X_train, Y_train)
predicted = gbr.predict(X_test)
residual = Y_test - predicted

fig = plt.figure(figsize=(30,30))
ax1 = plt.subplot(211)
sns.distplot(residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)

plt.title('Residual counts',fontsize=35)
plt.xlabel('Residual',fontsize=25)
plt.ylabel('Count',fontsize=25)

ax2 = plt.subplot(212)
plt.scatter(predicted, residual, color ='teal')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Predicted',fontsize=25)
plt.ylabel('Residual',fontsize=25)
plt.axhline(y=0)
plt.title('Residual vs. Predicted',fontsize=35)

plt.show()

print('Variance score: %.2f' % r2_score(Y_test, predicted))

rmse = np.sqrt(mean_squared_error(Y_test, predicted))
scores = cross_val_score(gbr, X, Y, cv=11)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)
#%%                default parameters for XGBosst
import xgboost as xgb

model = xgb.XGBRegressor()

model.fit(X_train,Y_train)
preds = model.predict(X_test)

from sklearn.metrics import r2_score
print (r2_score(Y_test, preds))
print('Variance score: %.2f' % r2_score(Y_test, preds))

rmse = np.sqrt(mean_squared_error(Y_test, preds))
scores = cross_val_score(model, X, Y, cv=12)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)
#%%              best parameters for XGBoost 

import xgboost as xgboost
best_xgb_model = xgboost.XGBRegressor(colsample_bytree=0.1,
                 gamma=0,                 
                 learning_rate=0.08,
                 max_depth=6,
                 min_child_weight=1,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=1,
                 subsample=1
                 )
best_xgb_model.fit (X_train, Y_train)
predicted = best_xgb_model.predict(X_test)
residual = Y_test - predicted
print('Variance score: %.2f' % r2_score(Y_test, predicted))

rmse = np.sqrt(mean_squared_error(Y_test, predicted))
scores = cross_val_score(best_xgb_model, X, Y, cv=12)

print('\nCross Validation Scores:')
print(scores)
print('\nMean Score:')
print(scores.mean())
print('\nRMSE:')
print(rmse)

#%%

sss=df['Seri'].value_counts()

for seri in sss.index:
    print("'"+seri+"'"+",") 


#%%
import numpy as np
user_input = {'Kilometre':100, 'Yil':2015, 'Vites':'Manuel', 'Yakit':'Benzin', 'Marka':'Hyundai', 'Seri':'i20'}
def input_to_one_hot(data):
    enc_input = np.zeros(339)  #132
    enc_input[0] = data['Yil']
    enc_input[1] = data['Kilometre']

    marks = df.Marka.unique()
    redefinded_user_input = 'Marka_'+data['Marka']
    mark_column_index = X.columns.tolist().index(redefinded_user_input)
    enc_input[mark_column_index] = 1

    fuel_types = df.Yakit.unique()
    redefinded_user_input = 'Yakit_'+data['Yakit']
    fuelType_column_index = X.columns.tolist().index(redefinded_user_input)
    enc_input[fuelType_column_index] = 1

    vites = df.Vites.unique()
    redefinded_user_input = 'Vites_'+data['Vites']
    vites_column_index = X.columns.tolist().index(redefinded_user_input)
    enc_input[vites_column_index] = 1
    
    seri = df.Seri.unique()
    redefinded_user_input = 'Seri_'+data['Seri']
    seri_column_index = X.columns.tolist().index(redefinded_user_input)
    enc_input[seri_column_index] = 1
    
    return enc_input

print(input_to_one_hot(user_input))
a = input_to_one_hot(user_input)
price_pred = gbr.predict([a])

print(price_pred[0])
#%%