#!/usr/bin/env python3
#Grid Search file

# In[2]:


import pandas as pd

featcsv = pd.read_csv("./train_feat.csv").fillna(0)

import numpy as np

featcsv["Type"]=np.where(featcsv["Type"]=="~",0,
                      np.where(featcsv["Type"]=="A",1,
                      np.where(featcsv["Type"]=="N",2,3)))


X = featcsv.values[:,2:]

y = featcsv["Type"].values

print("compute min max")
scaler = StandardScaler()
X = scaler.fit_transform(X)

dfObj = pd.DataFrame(X)
firstDups = dfObj[dfObj.duplicated()]
lastDups = dfObj[dfObj.duplicated(keep='last')]
duplicates = list(firstDups.index.values) + list(lastDups.index.values)

X = np.delete(X,list(duplicates),axis=0)
y = np.delete(y,list(duplicates),axis=0)

def report(results,classifier, n_top=3):
    file = open(classifier,"w") 
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            file.write("Model with rank: {0}".format(i)) 
            file.write("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate])) 
            file.write("Parameters: {0}".format(results['params'][candidate])) 
            file.write("\n\r") 

    file.close() 



print("running xgboost")

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
clf = XGBClassifier()
param_grid  = {"min_child_weight": [110,120,130],
             "max_depth": [380,390,400,410],
			 "gamma": [0],
			 "n_estimators": [100,110,120,130]}

grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
grid_search.fit(X, y)
report(grid_search.cv_results_,"xgboostGS.txt")


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

param_grid  = {"n_estimators": range(1800,1900,10),
              "max_depth": range(800,900,10)}

grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
grid_search.fit(X, y)
report(grid_search.cv_results_,"randomforestGS.txt")




from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

param_grid = {"n_neighbors": range(10,30,1),
              "algorithm": ["brute"],
              "leaf_size": range(1,20,1),
              "weights": ["distance"]}

grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
grid_search.fit(X, y)
report(grid_search.cv_results_,"nearestNGS.txt")


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()

param_grid = {"n_estimators": range(200,260,10),
              "max_depth":  range(10,50,5)}

grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
grid_search.fit(X, y)
report(grid_search.cv_results_,"boostGS.txt")