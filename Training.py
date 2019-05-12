#!/usr/bin/env python3
#Grid Search file
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in '%s': %s" % (cwd, files))
# In[2]:


import pandas as pd

print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)
featcsv = pd.read_csv("./train_feat.csv").fillna(0)

test = pd.read_csv("./test_signal.csv").fillna(0)
testFeatCsv = pd.read_csv("./test_feat.csv").fillna(0)
print("read in files...")


# Change the type to integers

# In[3]:


import numpy as np

train["Type"]=np.where(train["Type"]=="~",0,
                      np.where(train["Type"]=="A",1,
                      np.where(train["Type"]=="N",2,3)))


# In[4]:


X = featcsv.values[:,2:]
testX = testFeatCsv.values[:,1:]
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

def outputResults(output_data,filename):
    
    df = pd.DataFrame(output_data, columns = ['ID','Predicted'])
    df.to_csv(filename, index=None, header=True)

# In[42]:


print("running random forest + finding parameters")


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=800, n_estimators=1890)

clf.fit(X, y)
prediction = clf.predict(testX)


# In[44]:

prediction = np.where(prediction==0,'~',np.where(prediction==1,'A',np.where(prediction==2,'N','O')))
output_data = {'ID':list(testFeatCsv.values[:,0]),
                   'Predicted': list(prediction)}
outputResults(output_data,"forestResults.csv")


# In[41]:



from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=19,algorithm="brute", leaf_size=1,weights="distance")

clf.fit(X, y)
prediction = clf.predict(testX)
prediction = np.where(prediction==0,'~',np.where(prediction==1,'A',np.where(prediction==2,'N','O')))
output_data = {'ID':list(testFeatCsv.values[:,0]),
                   'Predicted': list(prediction)}
outputResults(output_data,"knnResults.csv")



