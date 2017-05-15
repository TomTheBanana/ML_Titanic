# -*- coding: utf-8 -*-
"""
Created on Wed May  3 21:12:21 2017

@author: thoma
"""


from sklearn.base import BaseEstimator, TransformerMixin

#==============================================================================
# ML transfrom functions
#==============================================================================
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
   
     

#%%
#==============================================================================
# Custom LabelBinarizer Transformer for multiple attributes
#==============================================================================
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelBinarizer

class MultiBinarizer(BaseEstimator, TransformerMixin):
    '''
    FIXME: labelbinarizer has problem with and integer array.
    quickfix-> first transform int to string in preparation
    '''
    def __init__(self):
         self.encoder = LabelBinarizer()
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        #create new list
        arrays = []     
        for attribute in X.T:
            #print("Attribute: ", attribute, " DataType: ", type(attribute[1]))
            arrays.append(self.encoder.fit_transform(attribute)) 
        return np.hstack(arrays)