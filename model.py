import cv2
import csv
import math
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.layers import Cropping2D
from keras.utils.visualize_util import plot



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

    # scale to 66x200x3 (same as nVidia)
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)

    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)

    images.append(image)
    measurement = float (line[3])
    measurements.append (measurement)


    #take the left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename

    imageBGR = cv2.imread(current_path)
    imageBGR = imageBGR[50:140,:,:]

    # scale to 66x200x3 (same as nVidia)
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)

    images.append(image)
    measurement = float (line[3]) + correction
    measurements.append (measurement)

    #take the right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/'+ filename

    imageBGR = cv2.imread(current_path)
    imageBGR = imageBGR[50:140,:,:]

    # scale to 66x200x3 for nVIDIA
    imageBGR = cv2.resize(imageBGR,(200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(imageBGR,cv2.COLOR_BGR2RGB)

    images.append(image)
    measurement = float (line[3]) - correctionRight
    measurements.append (measurement)


'''
Steering angle distribution adjustment. Do this to reduce the training bias of driving straight
'''

#visualize the distribution of angles for the collected images
num_bins = 23
avg_samples_per_bin = len(measurements)/num_bins
hist, bins = np.histogram(measurements, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.savefig("angle-distribution-before.png", bbox_inches="tight");



# the distribution angels are biased towards 0 degree, meaning a bias towards straight  driving.
# reduce the bias - determine the keep_probs
# if the number of samples is higher than avg_samples_per_bin
# bring the number of samples for that bin down to the average
keep_probs = []
target = avg_samples_per_bin * .5
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))
remove_list = []
for i in range(len(measurements)):
    for j in range(num_bins):
        if measurements[i] > bins[j] and measurements[i] <= bins[j+1]:
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
images = np.delete(images, remove_list, axis=0)
measurements = np.delete(measurements, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(measurements, num_bins)
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(measurements), np.max(measurements)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#plt.savefig("angle-distribution-after.png", bbox_inches="tight");



#data augmentation - Flipping Images And Steering Measurements
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)


'''
# comment out this portion while repeated training
# visualize how an image looks before augmentation
# comment out the steering angle distribution adjustment code above
fig = plt.figure(dpi=100, tight_layout=True, frameon=False) # dpi & figsize of my choosing
fig.figimage(images[1], cmap=plt.cm.binary)
fig.text(10.0,10.0,'image', size='medium', backgroundcolor='white', alpha=0.5)
plt.savefig("fig-sample-image[1].png")
plt.close(fig)

# visualize how an image looks after augmentation
# comment out the steering angle distribution adjustment code above
fig = plt.figure(dpi=100, tight_layout=True, frameon=False) # dpi & figsize of my choosing
fig.figimage(augmented_images[1], cmap=plt.cm.binary)
fig.text(10.0,10.0,'augmented image', size='medium', backgroundcolor='white', alpha=0.5)
plt.savefig("fig-aug-sample-image[1].png")
plt.close(fig)
'''


#now create the train set
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


'''
# split into train/test sets
image_train, image_validation = train_test_split(augmented_images, test_size=0.2)
measurements_train, measurements_validation = train_test_split(augmented_measurements, test_size=0.2)
print('Train:', image_train.shape, measurements_train.shape)
print('Validation:', image_validation.shape, measurements_validation.shape)
'''

#Train the network - NVIDIA model


model = Sequential()

# Normalize
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))

# Add three 5x5 convolution layers (output depth 24, 36, and 48), and 2x2 stride
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

#model.add(Dropout(0.50))

# Add two 3x3 convolution layers (output depth 64, and 64)
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Add a flatten layer
model.add(Flatten())

# Add three fully connected layers (depth 100, 50, 10) and dropouts
model.add(Dense(100, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.50))
model.add(Dense(50, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.50))
model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Dropout(0.50))

# Add a fully connected output layer
model.add(Dense(1))


model.compile(loss = 'mse', optimizer= 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)


model.save('model.h5')



#visualize the model
modelobj = load_model('model.h5')
plot (modelobj, to_file='model.png')
