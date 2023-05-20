import numpy as np
import os

import keras
from keras import layers
from keras import models

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

#load data
xData = np.load('images.npy')
yData = np.load('numerical_classifier.npy')

#data count
print("Loading Data\n")
f = open('dataCount.txt', 'r')
dataCount = int(f.read())


yData = to_categorical(yData) #convert to one-hot encoding
xData = xData.reshape((dataCount, 120, 320, 1)) #reshape to be the correct size
xData /= 255 #normalize

print("Data Loaded\n")

#Split data 80:10:10, train:validate:test
xTrain, xNext, yTrain, yNext = train_test_split(xData, yData, test_size = 0.2, random_state = 42)
xValidate, xTest, yValidate, yTest = train_test_split(xNext, yNext, test_size = 0.5, random_state = 42)

#create model
print("Creating Model\n")
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

print("Finished Creating Model\n")

print("Compiling Model\n")
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
print("Finished Compiling\n")

print("Fitting Model\n")
history = model.fit(xTrain, yTrain, epochs=5, batch_size=64, verbose=1, validation_data=(xValidate, yValidate))
print("Finished Fitting\n")

#plotting accuracy and loss
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Evaluating Model\n")
results = model.evaluate(xTest, yTest, batch_size=128)
print(results)

print("Saving Model\n")
model.save('model.h5')

print("Complete\n")
