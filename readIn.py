import pandas as pd
from keras.models import Sequential
from keras.layers import  Dense, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
import numpy as np

print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)

test = pd.read_csv("./test_signal.csv").fillna(0)
print("read in files...")
train.head()



train["Type"]=np.where(train["Type"]=="~",0,
                      np.where(train["Type"]=="A",1,
                      np.where(train["Type"]=="N",2,3)))


X = train.values[:,2:]


testX = test.values[:,1:]
y = train["Type"].values

print("data preprocessing...")
#https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/
dfObj = pd.DataFrame(X)
firstDups = dfObj[dfObj.duplicated()]
lastDups = dfObj[dfObj.duplicated(keep='last')]
duplicates = list(firstDups.index.values) + list(lastDups.index.values)


#print(list(firstDups.index.values))
X = np.delete(X,list(duplicates),axis=0)
y = np.delete(y,list(duplicates),axis=0)

def outputResults(output_data,filename):
    
    df = pd.DataFrame(output_data, columns = ['ID','Predicted'])
    df.to_csv(filename, index=None, header=True)

from keras.utils import to_categorical

y_train = to_categorical(y)

input_shape = (X.shape[1],1)