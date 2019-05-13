import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np

print("reading in files...")
train = pd.read_csv("./train_signal.csv").fillna(0)

test = pd.read_csv("./test_signal.csv").fillna(0)
print("read in files...")

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
print("Building model...")

epochs = 3
batch_size = 50

model = Sequential()

model.add(Conv2D(2, kernel_size=3, activation=’relu’))

model.add(Conv2D(2, kernel_size=3, activation='relu'))

model.add(Flatten())


#output layer
model.add(Dense(4, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X,y,epochs=epochs,batch_size=batch_size)

prediction = model.predict(testX)
print(prediction)