
# coding: utf-8

# Read in files
# fill in blank values

# In[2]:


import pandas as pd
from sklearn.svm import SVR
print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)
featcsv = pd.read_csv("./train_feat.csv").fillna(0)
print("read in files...")


# Change the type to integers

# In[3]:


import numpy as np

train["Type"]=np.where(train["Type"]=="~",0,
                      np.where(train["Type"]=="A",1,
                      np.where(train["Type"]=="N",2,3)))


# In[4]:


X = featcsv.values[:,2:]
y = train["Type"].values


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


# In[42]:


print("running random forest + finding parameters")
#
#from sklearn.model_selection import GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
#
#clf = RandomForestClassifier()
#
#param_grid = {"n_estimators": list(range(30,60)) + [100,200,500,1000],
#              "max_depth": list(range(5,20)) + [100,200,500,1000]}
#
#grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=10,verbose=2, cv=4)
#grid_search.fit(X, y)
#report(grid_search.cv_results_,"randomforest.txt")



# pip3 install keras tensorflow tensorflow-gpu numpy pandas sklearn

# In[41]:



#import heapq
#features = grid_search.best_estimator_.feature_importances_.tolist()
#print(features)
#bestFeaturesIndex = heapq.nlargest(10, xrange(len(features)), key=features.__getitem__)
#print(bestFeaturesIndex)


#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier()
#
#param_grid = {"n_neighbors": list(range(1,10)) + [20,50,100,200],
#              "algorithm": ["ball_tree","kd_tree","brute"],
#              "leaf_size": list(range(1,10)) + [20,50,100],
#              "weights": ["uniform","distance"]}
#
#grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=10,verbose=2, cv=4)
#grid_search.fit(X, y)
#report(grid_search.cv_results_,"nearestN.txt")


from sklearn.svm import SVC
clf = SVC()

param_grid = {"kernel": ["linear","poly","rbf", "sigmoid"]}

grid_search = GridSearchCV(clf, param_grid=param_grid,n_jobs=10,verbose=2, cv=4)
grid_search.fit(X, y)
report(grid_search.cv_results_,"SVM.txt")

