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
#Copy Data 
#==============================================================================
X = train_df.copy().drop("Survived", axis=1)
Label = train_df["Survived"].copy()


#==============================================================================
# Fillna
#==============================================================================
#fill missing categorical values with the most appearing category in the column
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].value_counts().index[0])
X["Sex"] = X["Sex"].fillna(X["Sex"].value_counts().index[0])
X["Pclass"] = X["Pclass"].apply(str)

#==============================================================================
# Categorise basic attributes
#==============================================================================
num_attribs = ["Age", "Fare", "Parch","SibSp"]
cat_attribs = ["Sex", "Embarked","Pclass"]

#%%
#==============================================================================
# Feature Engineering
#==============================================================================

#%%
#==============================================================================
# Cabin letter -- Not useful
#==============================================================================
'''
Take the first letter and all Nan filled with Z.
Randomforest 1000:
Accuracy:  0.790182442402
Precision:  0.732142857143
Recall:  0.719298245614     
'''
cabin = X["Cabin"]
cabin = cabin.fillna("Z")
cabinLetter = []
for cabinNum in cabin:
     cabinLetter.append(cabinNum[0])

X["Cabin"] = cabinLetter

#replace the rare T with Z to be compatible with the testset 
X["Cabin"] = X["Cabin"].replace('T', 'Z') 
 
cat_attribs.append("Cabin")
 
#%%
#==============================================================================
# Title 
#==============================================================================
'''
This feature shows that someone named miss is more likely to survive that 
a Mr. This information is already contained in the SEX feature.
It also shows that a Master or a Dr is more likely to survive than  Mr.

'''
X_title = train_df.copy()

X_title['Title'] = X_title["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(X_title['Title'], X_title['Survived'])
X_title[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()



#group rare titles to a rare category
X_title['Title'] = X_title['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X_title['Title'] = X_title['Title'].replace('Mlle', 'Miss')
X_title['Title'] = X_title['Title'].replace('Ms', 'Miss')
X_title['Title'] = X_title['Title'].replace('Mme', 'Mrs')
    
X_title[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#add this feature to training  and categorical attribute
X['Title'] = X_title['Title']

count = 0

if any("Title" in s for s in cat_attribs):   
     pass
else:
     cat_attribs.append("Title")



#%%
#TODO: LabelBinarizer with multiple input labels custom transformer
#==============================================================================
#Pipeline 
#==============================================================================
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from ml_transform import *

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', MultiBinarizer()),
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
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=4,n_jobs=-1, random_state = 42)
clf.fit(X_prepared, Label)


#%%
from sklearn.svm import SVC

clf = SVC(kernel='poly', degree=11, coef0=1, C=0.1)
clf.fit(X_prepared, Label)

#%%
#==============================================================================
# Ensemble Train
#==============================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


kn_clf = KNeighborsClassifier(n_neighbors =12, n_jobs=-1)
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier(n_estimators=10, min_samples_leaf=4,n_jobs=-1, random_state = 42)
svm_clf = SVC(kernel='poly', degree=12, coef0=100, C=0.1)

voting_clf = VotingClassifier(
 estimators=[('kn', kn_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')
voting_clf.fit(X_prepared, Label)

clf=voting_clf


#%%
#==============================================================================
# Grid Search
#==============================================================================
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

param_grid_svc = [
 {'degree': [11], 'coef0': [1,100], 'C': [1]}
 ]

param_grid_forest = [
 {'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}
 ]

clf = RandomForestClassifier(n_estimators = 1000)
#clf = SVC(kernel='poly')
grid_search = GridSearchCV(clf, param_grid_forest, cv=10, verbose=10, scoring='accuracy', n_jobs=-1)
#grid_search = GridSearchCV(clf, param_grid_svc, cv=10, verbose=10, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_prepared, Label)

#%%
#==============================================================================
# Cross-Validation Score
#==============================================================================
from sklearn.model_selection import cross_val_score, cross_val_predict

score = cross_val_score(clf, X_prepared, Label, cv=10 ,scoring="accuracy",n_jobs=-1)
print("Accuracy: ", score.mean())



#%%
#==============================================================================
# Precision Recall
#==============================================================================
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict

pred = cross_val_predict(clf, X_prepared, Label, cv=10)

precision = precision_score(Label, pred)
print("Precision: ", precision)

recall = recall_score(Label, pred)
print("Recall: ", recall)

#%%
#==============================================================================
#==============================================================================
# # TEST and EXPORT
#==============================================================================
#==============================================================================

X_Test = test_df.copy()

X_Test["Embarked"] = X_Test["Embarked"].fillna(X["Embarked"].value_counts().index[0])
X_Test["Sex"] = X["Sex"].fillna(X_Test["Sex"].value_counts().index[0])
X_Test["Pclass"] = X_Test["Pclass"].apply(str)

num_attribs = ["Age", "Fare", "Parch","SibSp"]
cat_attribs = ["Sex", "Embarked", "Pclass"]


cabin = X_Test["Cabin"]
cabin = cabin.fillna("Z")
cabinLetter = []
for cabinNum in cabin:
     cabinLetter.append(cabinNum[0])

X_Test["Cabin"] = cabinLetter
cat_attribs.append("Cabin")


X_title = test_df.copy()
X_title['Title'] = X_title["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

#group rare titles to a rare category
X_title['Title'] = X_title['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

X_title['Title'] = X_title['Title'].replace('Mlle', 'Miss')
X_title['Title'] = X_title['Title'].replace('Ms', 'Miss')
X_title['Title'] = X_title['Title'].replace('Mme', 'Mrs')
    
#add this feature to training  and categorical attribute
X_Test['Title'] = X_title['Title']
cat_attribs.append("Title")

X_test_prepared = preparation_pipeline.fit_transform(X_Test)


#%%
prediction= clf.predict(X_test_prepared)

#%%
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": prediction
    })
submission.to_csv( 'titanic_pred.csv' , index = False )
