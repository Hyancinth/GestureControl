import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#Load data
xData = np.load('xData.npy')
yData = np.load('yData.npy')


filePath =  './models/savedModel.h5' # Path to save model
 
#Callbacks
modelCheckpoint = ModelCheckpoint(filepath = filePath, save_best_only = True)
earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'auto', restore_best_weights = True, min_delta = 0)

#Split data 80:20, train:test
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42, stratify = yData)

print("xTrain shape:", xTrain.shape)
print("yTrain shape:", yTrain.shape)
print("xTest shape:", xTest.shape)
print("yTest shape:", yTest.shape)


print("Sample xTrain data:")
print(xTrain[0])  

print("Sample yTrain data:")
print(yTrain[0])  

print("Sample xTest data:")
print(xTest[0])  

print("Sample yTest data:")
print(yTest[0])  

#Create model
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(224, 224,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(Dropout(0.25, seed=21))
model.add(layers.Dense(5, activation='softmax'))

print(model.summary())

print("Compiling Model\n")
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training Model\n")
history = model.fit(xTrain, yTrain, epochs=200, batch_size=16, verbose=1, validation_data=(xTest, yTest), callbacks = [modelCheckpoint, earlyStopping])

#Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

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

#Classification report
yPred = model.predict(xTest)
yTrue = np.argmax(yTest, axis = 1)
report = classification_report(yTrue, np.argmax(yPred, axis = 1))
print(report)

print("Saving Model\n")
model.save(filePath)

print("Complete\n")
