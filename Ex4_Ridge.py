#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 21:52:21 2018

@author: pedrocampelo
"""


#Ridge Regression

#Com base de dados
import pandas as pd
import copy
detergent_df = pd.read_table('detergentyx.txt')

columns_x = list()
for column in detergent_df.columns:
    if('x' in column):
        columns_x.append(column)

X_df = copy.deepcopy(detergent_df[columns_x])
N, K = X_df.shape

fig, ax = plt.subplots()
ax.plot(np.linspace(1, K, K), X_df.transpose())
ax.set_xlim(1, K)
plt.xlabel("Frequencia")
plt.ylabel("Absorbancia") 
plt.show()


#Sem base da dados
from sklearn.datasets import load_boston
import pandas as pd   
import numpy as np

boston = load_boston()
print(boston.DESCR)

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret


#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
    
    
    
 #comentario 5   
    
    def hanoi (n, de, aux, para): 
        if n > 0: 
            hanoi (n-1, de, para, aux) 
            print ("Move de %i para %i" % (de, para))
            hanoi (n-1, aux, de, para) 

    if __name__ == '__main__': 
        print(hanoi(3, 1, 2, 3))

    
    
    
    
