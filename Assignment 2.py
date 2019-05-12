#!/usr/bin/env python3
#Random Search file

# In[2]:


import pandas as pd
print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)
featcsv = pd.read_csv("./train_feat.csv").fillna(0)

#test = pd.read_csv("./test_signal.csv").fillna(0)
#testFeatCsv = pd.read_csv("./test_feat.csv").fillna(0)
print("read in files...")


# Change the type to integers

# In[3]:


import numpy as np

train["Type"]=np.where(train["Type"]=="~",0,
                      np.where(train["Type"]=="A",1,
                      np.where(train["Type"]=="N",2,3)))

#test["Type"]=np.where(test["Type"]=="~",0,
#                      np.where(test["Type"]=="A",1,
#                      np.where(test["Type"]=="N",2,3)))
# In[4]:


X = featcsv.values[:,2:]
#testX = testFeatCsv.values[:,2:]
y = train["Type"].values
#testY = test["Type"].values

# In[17]:


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

#def outputResults(prediction):
#    df = DataFrame(prediction, columns = ['',''])
#    results.to_csv("submission.csv")

# In[42]:


print("running random forest + finding parameters")

#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier()
#
#param_dist  = {"n_estimators": range(10,5000,10),
#              "max_depth": range(10,5000,10)}

#grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
#grid_search.fit(X, y)
#report(grid_search.cv_results_,"randomforest.txt")

#random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                   n_iter=500, cv=4,random_state=64,n_jobs=-1,verbose=2)
#
#random_search.fit(X, y)
#report(random_search.cv_results_,"randomforest.txt")
# pip3 install keras tensorflow tensorflow-gpu numpy pandas sklearn

# In[41]:



#import heapq
#features = grid_search.best_estimator_.feature_importances_.tolist()
#print(features)


#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()
#
#param_dist = {"n_neighbors": range(1,40,1),
#              "algorithm": ["ball_tree","kd_tree","brute"],
#              "leaf_size": range(1,20,1),
#              "weights": ["uniform","distance"]}

#grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=-1,verbose=2, cv=4)
#grid_search.fit(X, y)
#report(grid_search.cv_results_,"nearestN.txt")


#random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                   n_iter=100, cv=4,random_state=64,n_jobs=-1,verbose=2)
#
#random_search.fit(X, y)
#report(random_search.cv_results_,"KNN.txt")
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()

param_dist = {"n_estimators": range(100,3000,10),
              "max_depth":  range(10,500,10)}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=250, cv=4,random_state=64,n_jobs=-1,verbose=2)

random_search.fit(X, y)
report(random_search.cv_results_,"boost.txt")
