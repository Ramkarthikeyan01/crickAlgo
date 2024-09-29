# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 23:26:36 2022

@author: 91790
"""
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from niapy.problems import Problem

class SVMFeat1(Problem):
    def __init__(self, X_train, y_train, para,alpha=0.99):
        super().__init__(dimension=X_train.shape[1], lower=0, upper=1)
        self.X_train = X_train
        self.y_train = y_train
        self.para=para
        self.alpha = alpha
        

    def _evaluate(self, x):        
        #selectedFeat = x > 0.5
        selectedFeat = x > 0.4
        #selectedFeat = x > self.para
        #print(selectedFeat)
        select_Count = selectedFeat.sum()        
        if select_Count == 0:
            return 1.0
        else:                        
            accuracy = cross_val_score(SVC(), self.X_train.iloc[:, selectedFeat], self.y_train, cv=2, n_jobs=-1).mean()
            score = 1 - accuracy
            num_features = self.X_train.shape[1]
            v1=self.alpha * score + (1 - self.alpha) * (select_Count / num_features)
            #print('acc = ',accuracy,' --- ',v1)            
            return v1