# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    # structure: dictionary of structure
    #            key is variable and value is parents of a variable
    # return list of learning order of variables 
    # ex) ['A', 'R', 'E', 'S', 'T', 'O']

    structure_list=[]
    for x,y in structure.items():
        if len(y) == 0 : #가장 부모가 되는 노드 찾기
            structure_list.append(x)
            
    for num in range(len(structure.keys())):
        for x,y in structure.items():
            set1 = set(y)                                                   # 자식 리스트와
            set2 = set(structure_list)                                      # 구조 리스트를 비교할 것
            if set1.intersection(set2) == set1 and x not in structure_list: # 자식 노드의 요소를 부모 노드가 모두 포함하고 있으면? 을 교집합을 사용해서 파악
                                                                            # 이미 구조 리스에 포함된 노드가 아닌 것만 구조 리스트에 넣음
                structure_list.append(x)

    return structure_list

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    new3 = {}

    for i in order1:
        new = []
        new2 = []
        new2.append(data[i])
        pa = structure.get(i)   # 해당 노드의 부모 구하기
        for j in pa:       
            new.append(data[j]) #부모 노드들의 데이터 값 추가
            
        if len(new)==0: # 부모노드가 없을때는 
            for c in order1: 
                aa = data[c].value_counts(normalize=True) # 확률 구해서
                df = pd.DataFrame(aa.values).T # 담기
                df.columns = aa.index # 열이름 설정
                new3[i]=df
        
        else: #있을때는
            final=pd.crosstab(new,new2 , normalize = True) #crosstab 기능을 통해 빈도수(확률) 행렬 생성
            new3[i]=final
   
    return new3
    
                
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        print(parms[var])
        #TODO: print the trained paramters
        
    
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}


order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')
str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}


order2=get_order(str2)
order2
parms2=learn_parms(data,str2,get_order(str2))
parms2
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')