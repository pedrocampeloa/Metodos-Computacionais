# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:56:32 2018

@author: Pedro
"""

#minha questao

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np

boston = load_boston()
print(boston.DESCR)

bh = pd.DataFrame(boston.data)
bh.columns = boston.feature_names
bh['PRICE'] = boston.target

print(bh.head())



 if __name__ == '__main__':

    import statsmodels.api as sm
    import patsy as ps

    y,X = ps.dmatrices('PRICE ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',data=bh, return_type='dataframe')

#OLS
    model = sm.OLS(y,X) 
    results = model.fit() 
    print (results.summary())
    
#MSE  
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

mseFull = np.mean((bh['PRICE'] - lr.predict(X))**2)
print(mseFull)


#LARS
from sklearn import linear_model

#1) Lars normal
 reg = linear_model.Lars(n_nonzero_coefs=500)
 reg.fit(X, y)
 print(reg.coef_) 
 
#2) Lasso Lars
reg2 = linear_model.LassoLars(alpha=0.01)
reg2.fit(X,y)    
print(reg2.coef_) 

#3) Lasso Lars com Cross Validation 
reg3 = linear_model.LassoLarsCV(cv=10)
reg3.fit(X,y) 
 print(reg3.coef_) 
 

#MSE
from sklearn import linear_model
import numpy as np

#1) Lars normal
mseLARS = np.mean((bh['PRICE'] - reg.predict(X))**2)
print(mseLARS)
print(reg.score(X,y))

#2) Lasso Lars
mseLARS2 = np.mean((bh['PRICE'] - reg2.predict(X))**2)
print(mseLARS2)
print(reg2.score(X,y))

#3) Lasso Lars com Cross Validation 
mseLARS3 = np.mean((bh['PRICE'] - reg3.predict(X))**2)
print(mseLARS3)
print(reg3.score(X,y))




                                                #questao felipe

    ### 1) IMPORTAÇÃO DOS DADOS ###

    from sklearn.datasets import load_boston

    boston = load_boston()
    print(boston.DESCR)

    import pandas as pd

    bh = pd.DataFrame(boston.data)
    bh.columns = boston.feature_names
    bh['PRICE'] = boston.target

    print(bh.head())

    ### 2) OLS ###

    from sklearn.linear_model import LinearRegression
    import numpy as np

    lr = LinearRegression()

    x = bh.drop('PRICE', axis=1)
    y = bh['PRICE']

    lr.fit(x, y)

    print('Coefficients: \n', lr.coef_)

    mseOLS = np.mean((bh['PRICE'] - lr.predict(x))**2)
    R2OLS = lr.score(x,y)
    print(mseOLS) ## MSE do modelo OLS ##
    print(R2OLS)  ## R² do modelo OLS ##

    ### 3) LARS ###

    import time
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LassoLarsCV
    from sklearn import linear_model

    ## Computing regularization path using the Lars lasso... ##
    t1 = time.time()
    model = LassoLarsCV(cv=10).fit(x, y)
    t_lasso_lars_cv = time.time() - t1



    # Display results
    m_log_alphas = -np.log10(model.cv_alphas_)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)

    plt.show()

    ## Computing LARS w/ alpha selected via CV (alpha = 0,01)... ##

    reg = linear_model.LassoLars(alpha=0.01)
    reg.fit(x, y)
    print('Coefficients: \n', reg.coef_) 

    mseLARS = np.mean((bh['PRICE'] - reg.predict(x))**2)
    R2LARS = reg.score(x,y)


    print(mseLARS) ## MSE do modelo LARS ##
    print(R2LARS) ## R² do modelo LARS ##


    ### 4) CROSS-VALIDATION OLS ###

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(x, bh.PRICE, test_size=0.33, 
    random_state=5)

    print (X_train.shape)
    print (X_test.shape)
    print (Y_train.shape)
    print (Y_test.shape)

    lm = LinearRegression()
    lm.fit(X_train, Y_train)


    print("Fit a model X_train, and calculate MSE with Y_train:", np.mean((Y_train - lm.predict(X_train))**2)) 
    print("Fit a model X_train, and calculate MSE with X_test, Y_test:", np.mean((Y_test - lm.predict(X_test)) ** 2) )



    print(lm.score(X_test, Y_test)) ### R² do OLS a partir dos dados 'fora da amostra' ###

    import matplotlib.pyplot as plt

    plt.scatter(lm.predict(X_train), lm.predict(X_train) - Y_train, c = 'b', s = 40, alpha = 0.5)   
    plt.scatter(lm.predict(X_test), lm.predict(X_test) - Y_test, c = 'g', s = 40)
    plt.hlines(y = 0, xmin=0,xmax=50)
    plt.title('Plotagem dos resíduos a partir dos dados de treinamento (azul) e teste 
    (verde)')
    plt.ylabel('Resíduos')
    plt.show()

    ### 5) CROSS VALIDATION LARS ###

    x2 = bh.drop(['INDUS', 'AGE', 'RAD', 'TAX'], axis = 1)

    X_train2, X_test2, Y_train2, Y_test2 = train_test_split(x2, bh.PRICE, test_size=0.33, 
    random_state=5)

    print (X_train2.shape)
    print (X_test2.shape)
    print (Y_train2.shape)
    print (Y_test2.shape)

    lm2 = LinearRegression()
    lm2.fit(X_train2, Y_train2)


    print("Fit a model X_train2, and calculate MSE with Y_train2:", np.mean((Y_train2 - 
    lm2.predict(X_train2))**2))
    print("Fit a model X_train2, and calculate MSE with X_test2, Y_test2:", 
    np.mean((Y_test2 - lm2.predict(X_test2)) ** 2) )


    print(lm2.score(X_test2, Y_test2)) ### R² do LARS a partir dos dados 'fora da amostra' 
    ###
    plt.scatter(lm2.predict(X_train2), lm2.predict(X_train2) - Y_train2, c = 'b', s = 40, alpha = 
    0.5)
    plt.scatter(lm2.predict(X_test2), lm2.predict(X_test2) - Y_test2, c = 'g', s = 40)
    plt.hlines(y = 0, xmin=0,xmax=50)
    plt.title('Plotagem dos resíduos a partir dos dados de treinamento (azul) e teste 
    (verde)')
    plt.ylabel('Resíduos')
    plt.show()


