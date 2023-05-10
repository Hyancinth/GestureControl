#imports 
from alive_progress import alive_bar
from time import sleep

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

#extract data from entire dataset
xData = []
yData = []
dataCount = 0
with alive_bar(10, title='Loading Data', bar='blocks') as bar: #progress bar
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
        sleep(0.02)            
        bar()
            
# xData = np.array(xData, dtype = 'float32')

# yData = np.array(yData)
# yData = yData.reshape(dataCount, 1) #reshape to be the correct size

print("\nLoading Data Complete\n")

#Save the data
# print("Saving Data...\n")

# np.save('images.npy', xData)
# np.save('numerical_classifier.npy', yData)

# print("Data Saved\n")

#save the data count
# with open('dataCount.txt', 'w') as f:
#     f.write(str(dataCount))
#     f.close()