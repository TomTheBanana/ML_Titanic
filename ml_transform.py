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