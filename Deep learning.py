filters =
k_size = 15
epochs = 8

print("begin building model...")
# In[3]:
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=input_shape))
#model.add(Dense(4, activation='softmax'))

model.add(Conv1D(filters, k_size, activation='relu'))
model.add(Conv1D(filters, k_size, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(filters, k_size, activation='relu'))
model.add(Conv1D(filters, k_size, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(filters, k_size, activation='relu'))
model.add(Conv1D(filters, k_size, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.4))

model.add(Dense(4, activation='softmax'))
print(model.summary())

# In[3]:
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
x_train_reshape = X.reshape(X.shape[0], X.shape[1],1)
model.fit(x_train_reshape, y_train, epochs=epochs)
# In[3]:
testX_reshape = testX.reshape(testX.shape[0],testX[1],1)
prediction = model.predict(testX_reshape)
predmax = np.argmax(prediction,1)

res = np.array(list("~ANO"))[predmax]

output_data = {'ID':list(test.values[:,0]),
                   'Predicted': list(res)}

outputResults(output_data,"cnnResults.csv")
# In[3]:
test_loss, test_acc = model.evaluate(x_train_reshape, y_train)
print('Test accuracy:', test_acc)