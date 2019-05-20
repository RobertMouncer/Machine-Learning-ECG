#!/import pandas as pd
print("reading in files...")
featcsv = pd.read_csv("./train_feat.csv").fillna(0)

print("read in files...")




import numpy as np

featcsv["Type"]=np.where(featcsv["Type"]=="~",0,
                      np.where(featcsv["Type"]=="A",1,
                      np.where(featcsv["Type"]=="N",2,3)))



X = featcsv.values[:,2:]

y = featcsv["Type"].values



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

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

param_dist  = {"n_estimators": range(10,5000,10),
             "max_depth": range(10,5000,10)}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                  n_iter=500, cv=4,random_state=64,n_jobs=-1,verbose=2)

random_search.fit(X, y)
report(random_search.cv_results_,"randomforest.txt")


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

param_dist = {"n_neighbors": range(1,40,1),
             "algorithm": ["ball_tree","kd_tree","brute"],
             "leaf_size": range(1,20,1),
             "weights": ["uniform","distance"]}


random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                  n_iter=100, cv=4,random_state=64,n_jobs=-1,verbose=2)

random_search.fit(X, y)
report(random_search.cv_results_,"KNN.txt")

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()

param_dist = {"n_estimators": range(100,3000,10),
              "max_depth":  range(10,500,10)}

random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=250, cv=4,random_state=64,n_jobs=-1,verbose=2)

random_search.fit(X, y)
report(random_search.cv_results_,"boost.txt")
