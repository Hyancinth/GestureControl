import numpy as np
import os

import keras
from keras import layers
from keras import models

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#load data
xData = np.load('images.npy')
yData = np.load('numerical_classifier.npy')

#data count
f = open('dataCount.txt', 'r')
dataCount = int(f.read())


yData = to_categorical(yData)
xData = xData.reshape((dataCount, 120, 320, 1))
xData /= 255

#Split data 80:10:10, train:validate:test
xTrain, xNext, yTrain, yNext = train_test_split(xData, yData, test_size = 0.2, random_state = 42)
xValidate, xTest, yValidate, yTest = train_test_split(xNext, yNext, test_size = 0.5, random_state = 42)

#create model
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

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(xTrain, yTrain, epochs=10, batch_size=64, verbose=1, validation_data=(xValidate, yValidate))

results = model.evaluate(xTest, yTest, batch_size=128)

model.save('model')