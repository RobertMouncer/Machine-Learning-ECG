import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv1D
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
def outputResults(output_data,filename):
    
    df = pd.DataFrame(output_data, columns = ['ID','Predicted'])
    df.to_csv(filename, index=None, header=True)
        
# In[3]:
from keras.utils import to_categorical

y_train = to_categorical(y)
print(train["Type"])
# In[3]:
print("Building model...")
input_shape = X[1].shape

model = Sequential()
model.add(Dense(units=20, activation='relu', input_shape=input_shape))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# In[3]:
model.fit(X, y_train, epochs=5)
# In[3]:
prediction = model.predict(testX)
predmax = np.argmax(prediction,1)

#predmax = np.where(predmax==0,'~',np.where(predmax==1,'A',np.where(predmax==2,'N','O')))
res = np.array(list("~ANO"))[predmax]

output_data = {'ID':list(test.values[:,0]),
                   'Predicted': list(res)}

outputResults(output_data,"knnResults.csv")