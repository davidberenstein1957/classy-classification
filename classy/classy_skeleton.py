from typing import Dict, List

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


class classySkeleton(object):
    def __init__(self, 
            name: str = 'text_categorizer',
            data: dict = None, 
            config: dict = {                              
                "C": [1, 2, 5, 10, 20, 100],
                "kernels": ["linear"],                              
                "max_cross_validation_folds": 5
            }
        ) -> None:
        self.name = name
        self.data = data
        self.config = config 
        
    def set_training_data(self, data: dict = None):
        if data: # update if overwritten
            self.data = data
            
        self.le = preprocessing.LabelEncoder()
        labels = []
        X = []
        self.label_list = list(self.data.keys())
        for key, value in self.data.items():
            labels += len(value)*[key]
            X += value
        self.y = self.le.fit_transform(labels)
        self.X = self.get_embeddings(X)
        
        if data: # update if overwritten
            self.set_svc()
            
    def get_embeddings(self, text):
        raise Exception("Not implemented for Skeleton, wrap it using a parent class.")
            
    def set_svc(self, config: dict = None):
        if config: # update if overwritten
            self.config = config
            
        C = self.config["C"]          
        kernels = self.config["kernels"]
        tuned_parameters = [{"C": C, "kernel": [str(k) for k in kernels]}]                                                                 
        folds = self.config["max_cross_validation_folds"]                       
        cv_splits = max(2, min(folds, np.min(np.bincount(self.y)) // 5))
        self.clf = GridSearchCV( 
            SVC(C=1, probability=True, class_weight='balanced'),
            param_grid=tuned_parameters, 
            n_jobs=1, 
            cv=cv_splits,         
            scoring='f1_weighted', 
            verbose=1
        )
        self.clf.fit(self.X, self.y)
        
    
        
        
