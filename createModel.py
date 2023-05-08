#imports 
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
from keras import layers
from keras import models
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

#load data
lookup = dict()
reverseLookup = dict()
count = 0

#traverse through the directory and create a dictionary of the labels
for i in os.listdir('data/leapGestRecog/00'):
    if not i.startswith('.'): #avoid hidden folders
        lookup[i] = count
        reverseLookup[count] = i 
        count = count + 1

# print(lookup,"\n")
# print(reverseLookup,"\n")

#extract data
print("Loading data...")
xData = []
yData = []
dataCount = 0
for i in range(0, 10):
    for j in os.listdir('data/leapGestRecog/0' + str(i) + '/'):
        if not j.startswith('.'): #avoid hidden folders
            count = 0 #count of images of a given gesture
            #loop over the images
            for k in os.listdir('data/leapGestRecog/0' + str(i) + '/' + j + '/'):
                #read in and convert to greyscale
                img = Image.open('data/leapGestRecog/0' + str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((320, 120))
                arr = np.array(img)
                xData.append(arr) 
                count = count + 1
            yValues = np.full((count, 1), lookup[j]) 
            yData.append(yValues)
            dataCount = dataCount + count
            
xData = np.array(xData, dtype = 'float32')
yData = np.array(yData)
yData = yData.reshape(dataCount, 1) #reshape to be the correct size

print("Done loading data")
