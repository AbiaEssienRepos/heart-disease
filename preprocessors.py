import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class MeanImputation():
    """Performs mean imputation for zero values"""
    
    def __init__(self,variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
            
        self.variables = variables
        self.params_ = {}
        
    def fit(self, X, y=None):
        
        for variable in self.variables:
            tmp = X[X[variable]!=0]
            mean = tmp[variable].mean()
            self.params_[variable] = mean
            
        return self
    
    def transform(self, X, y=None):
        
        for variable in self.variables:
            X[variable] = np.where(X[variable]!=0, X[variable], self.params_[variable])
        
        return X
    
    
class CategoricalEncoder():
    """Performs one hot encoding on categorical variables"""
    
    def __init__(self, variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
            
        self.variables = variables
        self.encoder_dict_ = {}
        
    def fit(self, X, y=None):
        # persist column and dummy columns pair in dictionary
        
        X[self.variables] = X[self.variables].astype(str)
        
        for feature in self.variables:
            dummies = pd.get_dummies(X[feature],drop_first=True)
            for column in dummies.columns:
                dummies = dummies.rename(columns={column:feature + '_' + column})
            self.encoder_dict_[feature] = list(dummies.columns)
            
        return self
            
    def transform(self, X, y=None):
        
        X[self.variables] = X[self.variables].astype(str)
        
        for feature in self.variables:
            dummies = pd.get_dummies(X[feature],drop_first=True)
            for column in dummies.columns:
                dummies = dummies.rename(columns={column:feature + '_' + column})
            
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(feature, axis=1)
            
        return X
    
    
class OrdinalEncoder():
    """Performs ordinal encoding on non-binary variables"""
    
    def __init__(self, variables, target):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
            
        self.variables = variables
        self.params_ = {'ChestPainType': {},
                        'RestingECG': {},
                        'ST_Slope': {}}
        self.target = target
        self.ordinal_labels_ = {}
        
    def fit(self, X, y):
        
        X = pd.concat([X,y], axis=1)
        
        for variable in self.variables:
        
            for label in X[variable].unique():
                label_disease = len(X[(X[variable]==label) & (X[self.target]==1)])
                label_size = len(X[X[variable]==label])
                self.params_[variable][label] = label_disease / label_size
                
        return self
                
    def transform(self, X, y=None):
        
        for variable in self.variables:
            
            labels = pd.Series(self.params_[variable])
            ordered_labels = labels.sort_values().index
            ordinal_label = {k: i for i, k in enumerate(ordered_labels, 1)}
            
            self.ordinal_labels_[variable] = ordinal_label
    
            X[variable] = X[variable].map(ordinal_label)
        
        return X
    
    
class ContinuousScaler():
    """Scales and returns a chosen subset of continuous variables"""
    
    def __init__(self, variables):
        
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
            
        self.variables = variables
        
    def fit(self, X, y=None):
        # learn and persist the mean and standard deviation
        # of the dataset
        
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(X[self.variables])
        return self
        
    def transform(self, X, y=None):
        
        X[self.variables] = self.scaler_.transform(X[self.variables])
        return X