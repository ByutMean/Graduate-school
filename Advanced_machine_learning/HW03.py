# -*- coding: utf-8 -*-
# DO NOT CHANGE
import numpy as np
from itertools import product
from sklearn.svm import OneClassSVM
from scipy.sparse.csgraph import connected_components
import pandas as pd
import matplotlib.pyplot as plt

def get_adj_mat(X,svdd,num_cut):
    # X: n*p input matrix
    # svdd: trained svdd model by sci-kit learn using X
    # num_cut: number of cutting points on line segment
    #######OUTPUT########
    # return adjacent matrix size of n*n (if two points are connected A_ij=1)
    
    svdd.fit(X)
    sv = svdd.support_[svdd.dual_coef_[0] != 1]
    adj_matrix = np.zeros((len(X), len(X))) # X by X 사이즈의 영행렬 생성
    prod = list(product(range(len(X)), range(len(X)))) #product를 이용해 (0,0) 부터 (len(X),len(X))까지 생성
    
    for i, j in prod : #i는 행 j는 열이므로
        if i == j : #행과 열이 같으면
           adj_matrix[i][j] = 0 
            
        else :
            if (svdd.decision_function(np.linspace(X[i], X[j],num_cut))  # 기존의 거리를 측정한 값 num_cut번 반복한 모든 값(a)이
                >= np.min(svdd.decision_function(X[sv]))).sum() == num_cut: # 거리의 최소값(b)보다 크거나 같으면 => (b) 이상인 (a)의 개수가 num_cut과 같으면 
                adj_matrix[i][j] = 1 # 연결
                
            else :
                adj_matrix[i][j] = 0 # 아니면 연결 안함
                
    return adj_matrix #행렬 반환
    
    
    
def cluster_label(A,bsv):
    # A: adjacent matrix size of n*n (if two points are connected A_ij=1)
    # bsv: index of bounded support vectors
    #######OUTPUT########
    # return cluster labels (if samples are bounded support vectors, label=-1)
    # cluster number starts from 0 and ends to the number of clusters-1 (0, 1, ..., C-1)
    # Hint: use scipy.sparse.csgraph.connected_components
    
    labels = np.arange(len(A)) # 라벨링할 것 = 인접행렬 길이만큼 생성
    matrix = A[np.setdiff1d(np.arange(len(A)), bsv),:][:,np.setdiff1d(np.arange(len(A)), bsv)]
             # 라벨 - bounded => 차집합으로 non 와 unbounded 만 남김
    num_of_comp, label = connected_components(matrix)
    #클러스터 수, 클러스터라벨
    
    for i in range(len(labels)): 
        if i in bsv: # bounded support vector에 속하면
            labels[i] = -1 # -1 클러스터로 라벨링
        else: # 아닌 경우
            labels[i] = label[0] # connected_components로 계산된 라벨값 할당
            label = np.delete(label, 0) #첫번째 라벨값 제거(다음으로 넘기기 위함)

    return labels




ring=pd.read_csv('https://drive.google.com/uc?export=download&id=1_ygiOJ-xEPVSIvj3OzYrXtYc0Gw_Wa3a')
num_cut=20
svdd=OneClassSVM(gamma=1, nu=0.2)


X = ring.values 
adj = get_adj_mat(X, svdd, num_cut)

bsv = svdd.support_[svdd.dual_coef_[0] == 1] # 1이면 bounded
ubsv = svdd.support_[svdd.dual_coef_[0] != 1] #1이 아니면 unbounded
nsv = np.setdiff1d(np.arange(len(ring)), np.append(bsv, ubsv)) # 전체에서 bounded, unbounded 차집합한 값은 non

clus_label = cluster_label(adj, bsv) #라벨링



##########Plot1###################
# Get SVG figure (draw line between two connected points with scatter plots)
# draw decision boundary
# mark differently for nsv, bsv, and free sv


#plot1

xx, yy = np.meshgrid(np.linspace((ring['X1'].min() - 0.5), (ring['X1'].max() + 0.5), 100), 
                     np.linspace((ring['X2'].min() - 0.5), (ring['X2'].max() + 0.5), 100))
zz = np.c_[xx.ravel(), yy.ravel()]
zz_pred = svdd.decision_function(zz)


plt.contour(xx,yy,zz_pred.reshape(xx.shape), levels=[0], linewidth=5, colors = 'purple') # 클러스터 영역 표시
plt.scatter(ring['X1'][bsv], ring['X2'][bsv], marker='x',color = 'blue') # bounded는 x표시
plt.scatter(ring['X1'][ubsv], ring['X2'][ubsv], marker='o',color = 'red', facecolors='none') #unbounded는 빨간o표시
plt.scatter(ring['X1'][nsv], ring['X2'][nsv], marker='o',color = 'black') # nonbounded는 검은 o표시 


# 선 그리기
prod = list(product(np.arange(len(X)),np.arange(len(X))))
for i,j in prod:
    if (i in bsv) or (j in bsv):
        continue
    if adj[i][j] == 1:
        x_list = [X[i,0], X[j,0]]
        y_list = [X[i,1], X[j,1]]
        plt.plot(x_list, y_list , linewidth=0.3, color = 'black')

plt.show()
plt.close()

##########Plot2###################
# Clsuter labeling result
# different clusters should be colored using different color
# outliers (bounded support vectors) are marked with 'x'


plt.contour(xx, yy, zz_pred.reshape(xx.shape), levels=[0], colors = 'purple')
plt.scatter(ring['X1'][bsv], ring['X2'][bsv], marker='x', color = 'blue') # bounded 표시
plt.scatter(X[clus_label == 0,0], X[clus_label == 0,1], color = '#660033')
plt.scatter(X[clus_label == 1,0], X[clus_label == 1,1], color = '#1245AB')
plt.scatter(X[clus_label == 2,0], X[clus_label == 2,1], color = '#47C83E')
plt.scatter(X[clus_label == 3,0], X[clus_label == 3,1], color = '#FAED7D') 
#각 클러스터별 scatter plot 그리기


