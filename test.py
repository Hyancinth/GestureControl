import numpy as np
from tensorflow import keras
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

filePath =  './models/savedModel.h5'
model = keras.models.load_model(filePath)

img = cv2.imread('a.jpeg', cv2.IMREAD_COLOR)
img = cv2.flip(img, 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (41, 41), 0)
ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thresh = cv2.resize(thresh, (224, 224))
arr = np.array(thresh, dtype = 'float32')
arr = arr.reshape((1, 224, 224, 1))
arr = arr/255

gesture_names = {
                 0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

prediction = model.predict(arr)
predict = gesture_names[np.argmax(prediction)]
print(prediction)
print(predict)

