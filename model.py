import cv2
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.layers import Cropping2D


lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []
correction = 0.2
correctionRight = 0.2

for line in lines:

    #take the center image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename

    imageBGR = cv2.imread(current_path)
    imageBGR = imageBGR[50:140,:,:]
    # apply subtle blur
    imageBGR = cv2.GaussianBlur(imageBGR, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2YUV)
    images.append(image)
    measurement = float (line[3])
    measurements.append (measurement)


    #take the left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename

    imageBGR = cv2.imread(current_path)
    imageBGR = imageBGR[50:140,:,:]
    # apply subtle blur
    imageBGR = cv2.GaussianBlur(imageBGR, (3,3), 0)
    # scale to 66x200x3 (same as nVidia)
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2YUV)
    images.append(image)
    measurement = float (line[3]) + correction
    measurements.append (measurement)

    #take the right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename

    imageBGR = cv2.imread(current_path)
    imageBGR = imageBGR[50:140,:,:]
    # apply subtle blur
    imageBGR = cv2.GaussianBlur(imageBGR, (3,3), 0)
    # scale to 66x200x3 for nVIDIA
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2YUV)
    images.append(image)
    measurement = float (line[3]) - correctionRight
    measurements.append (measurement)


#data augmentation - Flipping Images And Steering Measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)



#now create the train set
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)




#Train the network
ch, row, col = 3, 66, 200

model = Sequential()
#preprocessing - normalization and mean centering
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))


model.compile(loss = 'mse', optimizer= 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)


model.save('model.h5')
