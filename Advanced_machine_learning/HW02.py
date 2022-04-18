# -*- coding: utf-8 -*-
# Use the following packages only
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


# implement consine kernel
def cosine_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    u_set = [(x - i) / h for i in train]
    result = []


    for i in u_set:
        if np.abs(i) <= 1:
            func_u = (np.pi/4) * (np.cos(np.pi/2 * i))
        else : 
            func_u = 0
        result.append(func_u)
        
        
    px = (1/len(train) * np.sum(np.multiply((1/h), result)))
    return px


# implement gaussian kernel
def gaussian_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    u_set = [(x - i) / h for i in train]
    result = []
    
    
    for i in u_set:
        temp = norm.pdf(i)
        result.append(temp)
    
    
    px = (1/len(train) * np.sum(np.multiply((1/h), result)))
    return px

                                  
# implement 2D gaussian kernel
def gaussian_2d_kernel(x, train, h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    u_set = [(x - i) / h for i in train]

    
    mean = np.array([0,0])
    result = multivariate_normal.pdf(u_set, mean = mean)
    
    
    px = (1/len(train) * np.sum(np.multiply((1/h**2), result)))
    return px 


# implement epanechnikov kernel
def epanechnikov_kernel(x,train,h):
    # x: test point
    # train: training set
    # h: bandwidth
    # return p(x)
    
    u_set = [(x - i) / h for i in train]
    result = []
    
    
    for i in u_set:
        if np.abs(i) <= 1:
            func_u = (3/4) * (1 - i**2)
        else : 
            func_u = 0
        result.append(func_u)
        
        
    px = (1/len(train) * np.sum(np.multiply((1/h), result)))
    return px


def kde1d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    # return d, d contains probaiblity density values of test samples (1d array)
    
    d = [kernel(x,train,h) for x in test]
    
    return d

def kde2d(train,test,kernel,h):
    # train: training set
    # test: test set
    # kernel: kernel function (object)
    # h: bandwidth
    # return d, d contains probaiblity density values of test samples (1d array)
    
    d = [kernel(x, train, h) for x in test]
    
    return d

# 1D
sample=[2,3,4,8,10,11,12]
h=1
x=np.linspace(0,14,100000)

y1=kde1d(sample,x,cosine_kernel,h)
y2=kde1d(sample,x,gaussian_kernel,h)
y3=kde1d(sample,x,epanechnikov_kernel,h)
    
fig=plt.subplots(1,3,figsize=(10,4))
plt.subplot(1,3,1)
plt.plot(x,y1)
plt.title('Cosine')
plt.subplot(1,3,2)
plt.plot(x,y2)
plt.title('Gaussian')
plt.subplot(1,3,3)
plt.plot(x,y3)
plt.title('Epanechnikov')
plt.show()

#2D
sample_2d=pd.read_csv(r'https://drive.google.com/uc?export=download&id=1uyPHjquXOIS9TTrG9Nb_gW3sfQEOdY0V')
sum_stats=sample_2d.describe()
xmin,ymin=sum_stats.loc['min']-0.5
xmax,ymax=sum_stats.loc['max']+0.5

x=np.linspace(xmin,xmax,100)
y=np.linspace(ymin,ymax,100)
X,Y=np.meshgrid(x,y)
Z = np.c_[X.ravel(),Y.ravel()]

Z1 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,0.5)
Z1 = np.reshape(Z1, X.shape)
Z2 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,1)
Z2 = np.reshape(Z2, X.shape)
Z3 = kde2d(sample_2d.values,Z,gaussian_2d_kernel,2)
Z3 = np.reshape(Z3, X.shape)

fig,ax=plt.subplots(1,3,figsize=(16,4))
plt.subplot(1,3,1)
cs=plt.contourf(X,Y,Z1,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.subplot(1,3,2)
cs=plt.contourf(X,Y,Z2,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.subplot(1,3,3)
cs=plt.contourf(X,Y,Z3,cmap=plt.cm.Blues)
plt.colorbar(cs)
plt.show()
