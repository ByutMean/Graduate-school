# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    # X: inpute variables
    # y: target

    n = X.shape[0]
    p = X.shape[1]

    
    matrix_one = (np.ones(n).reshape(-1,1))
    X2 = np.concatenate((matrix_one, X), axis=1)
    y2 = y.reshape(-1,1)


    XTX = np.matmul(X2.T, X2)
    XTX_inv = np.linalg.inv(XTX)
    beta_hat = np.matmul(np.matmul(XTX_inv, X2.T), y2)
    y_hat = np.matmul(X2, beta_hat)


    SSR = sum((y_hat - np.mean(y2))**2)
    SSE = sum((y_hat - y2)**2)
    SST = SSR + SSE
    MSR = SSR / p
    MSE = SSE / (n - p - 1)
    
    
    f_value = MSR / MSE
    p_value = 1- stats.f.cdf(f_value, p, n - p - 1)


    print('==========================================================================================')
    print('Factor          SS            DF          MS              F-value            pr>F')
    print('Model   ',SSR, '   ', p, ' ', MSR, '  ', f_value, '  ', p_value) 
    print('Error   ',SSE, '   ', n - p - 1, ' ', MSE)
    print('------------------------------------------------------------------------------------------')
    print('Total   ',SSE + SSR, '   ', p + n - p - 1)
    print('==========================================================================================')
    
    
    return 0


def ttest(X,y,varname=None):
    # X: inpute variables
    # y: target
    
    n = X.shape[0]
    p = X.shape[1]

    
    matrix_one = (np.ones(n).reshape(-1,1))
    X2 = np.concatenate((matrix_one, X), axis=1)
    y2 = y.reshape(-1,1)


    XTX = np.matmul(X2.T, X2)
    XTX_inv = np.linalg.inv(XTX)
    beta_hat = np.matmul(np.matmul(XTX_inv, X2.T), y2)
    y_hat = np.matmul(X2, beta_hat)


    SSR = sum((y_hat - np.mean(y2))**2)
    SSE = sum((y_hat - y2)**2)
    SST = SSR + SSE
    MSR = SSR / p
    MSE = SSE / (n - p - 1)
    
    
    se2 = MSE * XTX_inv
    se2_diagonal = np.diag(se2)
    se = np.sqrt(se2_diagonal)

 
    t_value = [beta_hat[i] / np.sqrt(se2_diagonal[i]) for i in range(len(se2_diagonal))]
    p_value = ((1 - stats.t.cdf(np.abs(np.array(t_value)), n - p - 1)) * 2)
    
    
    name = np.append('const',data.feature_names)
    print('==========================================================================================')
    print('Variable     coef                 se                  t                 Pr>|t|')
    for i in range(len(name)):
        print(name[i],'   ',beta_hat[i],'   ',se[i],'   ',t_value[i],'    ',p_value[i])
    print('==========================================================================================')
    
    
    return 0

## Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)
