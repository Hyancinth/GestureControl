import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from keras.utils import to_categorical

gestures = {'L_': 'L',
           'fi': 'Fist',
           'C_': 'C',
           'ok': 'Okay',
           'pe': 'Peace',
           'pa': 'Palm'
            }

gestures_map = {'Fist' : 0,
                'L': 1,
                'Okay': 2,
                'Palm': 3,
                'Peace': 4
                }


def processImage(path):
    img = Image.open(path)
    img = img.resize((224, 224))
    img = np.array(img)
    return img

def processData(xData, yData):
    xData = np.array(xData, dtype = 'float32')
    xData /= 255 #normalize
    yData = to_categorical(yData)
    return xData, yData

def traverseFileTree(path):
    xData = []
    yData = [] 
    for directory, subdirectories, files in os.walk(path):
        for file in files:
            if (not file.startswith('.')) and (not file.startswith('C_')):
                path = os.path.join(directory, file)
                gesture = gestures[file[0:2]]
                img = processImage(path)
                xData.append(img)
                yData.append(gestures_map[gesture])
            else:
                continue
    
    xData, yData = processData(xData, yData)
    return xData, yData

def saveData(xData, yData):
    np.save('xData.npy', xData)
    np.save('yData.npy', yData)
    


def main():
    print("Loading Data...\n")
    xData, yData = traverseFileTree('frames/silhouettes/')
    print("Complete\n")
    
    xData = xData.reshape(xData.shape[0], 224, 224, 1) #reshape to be the correct size
    
    print(f'X_data shape: {xData.shape}')
    print(f'y_data shape: {yData.shape}')
    
    plt.imshow(xData[0], cmap = 'gray')
    plt.show()
    
    print("Saving Data...\n")
    saveData(xData, yData)

if __name__ == '__main__':
    main()