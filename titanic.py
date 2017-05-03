# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#==============================================================================
# import csv
#==============================================================================
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")


#%%
#==============================================================================
#Prepare Data 
#==============================================================================
X = train_df.copy().drop("Survived", axis=1)
Label = train_df["Survived"].copy()


#fill missing categorical values with the most appearing category in the column
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].value_counts().index[0])
X["Sex"] = X["Sex"].fillna(X["Sex"].value_counts().index[0])


num_attribs = ["Age"]
cat_attribs = ["Sex", "Embarked"]


#%%
#TODO: LabelBinarizer with multiple input labels custom transformer
#==============================================================================
#Pipeline 
#==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from ml_transform import *

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])

preparation_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

X_prepared = preparation_pipeline.fit_transform(X)

#%%
#==============================================================================
# Train
#==============================================================================




#%%
#==============================================================================
# Cross-Validation
#==============================================================================
