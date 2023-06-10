import cv2
import numpy as np
from keras.models import load_model
import os

prediction = ""

gestureNames = {
                 0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

filePath = os.path.join(os.curdir, 'models', 'savedModelV5.h5')
model = load_model(filePath)

def getPrediction(img):
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (41, 41), 0)
    ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.resize(thresh, (224, 224))
    arr = np.array(thresh, dtype = 'float32')
    arr = arr.reshape((1, 224, 224, 1))
    arr = arr/255
    
    prediction = model.predict(arr)
    predict = gestureNames[np.argmax(prediction)]
    score = float("%0.2f" % (max(prediction[0]) * 100))
    print(prediction)
    
    return predict, score

cap = cv2.VideoCapture(0)
cap.set(10, 200)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    # frame = cv2.bilateralFilter(frame, 5, 50, 100)
    
    prediction, score = getPrediction(frame)
    # action = gestureNames[np.argmax(prediction)]

    cv2.putText(frame, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255))
    # cv2.putText(frame, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255))  # Draw the text
    
    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break

cv2.destroyAllWindows()
cap.release()