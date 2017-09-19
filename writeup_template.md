**Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 showing the driving of the car in track 1
* im2video.py file that was used to generate video from the run images
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file. The drive.py file was modified to reflect some preprocessing of the image data that were performed outside the model in the model.py file. Those preprocessing include cropping and bluring images. No normalization was done inside drive.py as the normalization was done inside the model in model.py file. The car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the suggested NVIDIA model as referred in the class

The model includes RELU layers to introduce nonlinearity (code line 109), and the data is normalized in the model using a Keras lambda layer (code line 107).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 114, 117 etc.).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with using a simple one layer model, trained the network and was able to run the car a little while.
The loss on those training was greater than 2. Then I preprocessed the images inside the model by normalizing and mean centering (model.py line 107).

Although the model improves, it was not enough. So I move to a more complex model - LeNet.
I then use data augmentation by Flipping Images and Steering Measurements (model.py line 86-91)

So far the model was trained using only the center images. Later I included more images - that is
left and right camera images to better train the model. At this point, the car was driving much better
but still not got enough. The car was more prone to be close to the left lane as the training dataset
had more images on the left. So I used a correction factor (correction and correctionRight parameter in model.py)
parameter to keep the car closer to the center lane of the road.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

I removed the unnecessary scenes from the images that are not part of the road by cropping
the top and bottom portions of the data set.

Finally, I used even more complex model as suggested by Udacity - the [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

To combat the overfitting, I modified the model to include dropout.



####2. Final Model Architecture

The final model architecture I used is the [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)



####3. Creation of the Training Set & Training Process

I used the training data set provided by Udacity.

To read the data, I used cv2.imread which returns images in BGR format while drive.py sends images in RGB format.
So in model.py, after reading the images using cv2.imread, I convert them to RGB format.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as after 5, the loss seem to increase. I used an adam optimizer so that manually training the learning rate wasn't necessary.


####4. Thoughts and Learnings
I used NVIDIA model. Another option can be to use [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py)

This [helpful guide](https://slack-files.com/T2HQV035L-F50B85JSX-7d8737aeeb) from Paul was useful
specially in terms of avoiding the RGB and BGR pitfalls.

Another helpful guide on changes that need to be applied to drive.py is [here](https://discussions.udacity.com/t/help-car-does-not-move-when-implementing-the-model/242563/13?u=subodh.malgonde)

Note to self: spend more time with [generators](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/46a70500-493e-4057-a78e-b3075933709d/concepts/b602658e-8a68-44e5-9f0b-dfa746a0cc1a)

I had some issues with keras version. Always make sure you are using the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/environment.yml)

Using the Udacity provided video.py file, I couldn't generate the mp4 video. So I used im2video.py file
found [here](http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html)to generate video from images. 
