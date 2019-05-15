import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout
import tensorflow as tf
import numpy as np

print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)

test = pd.read_csv("./test_signal.csv").fillna(0)
print("read in files...")
train.head()

# In[3]:

train["Type"]=np.where(train["Type"]=="~",0,
                      np.where(train["Type"]=="A",1,
                      np.where(train["Type"]=="N",2,3)))

# In[4]:
X = train.values[:,2:]


testX = test.values[:,1:]
y = train["Type"].values

# In[4]:
#https://thispointer.com/pandas-find-duplicate-rows-in-a-dataframe-based-on-all-or-selected-columns-using-dataframe-duplicated-in-python/
dfObj = pd.DataFrame(X)
firstDups = dfObj[dfObj.duplicated()]
lastDups = dfObj[dfObj.duplicated(keep='last')]
duplicates = list(firstDups.index.values) + list(lastDups.index.values)

# In[4]:
#print(list(firstDups.index.values))
X = np.delete(X,list(duplicates),axis=0)
y = np.delete(y,list(duplicates),axis=0)
# In[4]:
def outputResults(output_data,filename):
    
    df = pd.DataFrame(output_data, columns = ['ID','Predicted'])
    df.to_csv(filename, index=None, header=True)
        
# In[3]:
from keras.utils import to_categorical

y_train = to_categorical(y)
# In[3]:
print("Building model...")
input_shape = (X.shape[1],1)
print(input_shape)
print(X.shape)
# In[3]:
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=input_shape))
#model.add(Dense(4, activation='softmax'))

model.add(Conv1D(20, 10, activation='relu'))
model.add(Conv1D(30, 10, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(40, 10, activation='relu'))
model.add(Conv1D(20, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(20))
model.add(Dropout(0.4))

model.add(Dense(4, activation='softmax'))
print(model.summary())
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# In[3]:
x_train_reshape = X.reshape(X.shape[0], X.shape[1],1)
model.fit(x_train_reshape, y_train, epochs=5)
# In[3]:
prediction = model.predict(testX)
predmax = np.argmax(prediction,1)

res = np.array(list("~ANO"))[predmax]

output_data = {'ID':list(test.values[:,0]),
                   'Predicted': list(res)}

outputResults(output_data,"cnnResults.csv")
# In[3]:
test_loss, test_acc = model.evaluate(X, y_train)
print('Test accuracy:', test_acc)