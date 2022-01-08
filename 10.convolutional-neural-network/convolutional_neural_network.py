import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)

#Preprocessing the training set
#we need to apply some transformation to training set in order to avoid overfitting
#Some geometrical transformations which is called image augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#rescale is for feature scaling. Each pixel takes a value from 0 and 255. so dividing it by 255 will produce a 
#value between 0 and 1
training_set = train_datagen.flow_from_directory(
    '10.convolutional-neural-network/dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
#feature scaling for test set, but don't apply augmentation(transformation)
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    '10.convolutional-neural-network/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
#Building the CNN
cnn = tf.keras.models.Sequential()

#Step-1 Convolution
#filters correspond to the initial set of layers for convolution
#kernel size represents the feature detector size
#activation function can be rectifier since this is not an output layer
#input shape is 64, 64 because in datagen we reshaped to 64, 64. 3 is used since the image is coloured
#if the input images are black and white we can use 1 instead of 3
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

#step-2 pooling
#strides indicate by how many pixel, we are shifting the slide 
#we are moving by 2 pixels and the slide is 2, 2 size
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Adding a second convolutional layer
#copy pasting the first two steps here, but we  need to change the input_shape parameter because, this layer might not get the
#input in 64, 64 size
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#Step-3 Flattening
cnn.add(tf.keras.layers.Flatten())

#Step-4 Full connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Step 5- Output layer
#we need binary output, only cat or dog, so output layer will have only 1 neuron
#For output layer, we will have sigmoid activation function since we are doing binary classification
#If we were doing multi class classification, we need to provide softmax actication function
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the CNN on training set and evaluating on test set
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

#Making a single prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('10.convolutional-neural-network/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
#This generates a test image in PIL format

#The below line converts the image from PIL format to an array which necessary for the prediction
test_image = image.img_to_array(test_image)

#Also when we were training the dataset, we didn't train on a single image but with a batch of 32 images which is an 
#extra dimension, so for prediction also, we need to convert the input image to 32 batches
test_image = np.expand_dims(test_image, axis=0)

result = cnn.predict(test_image)
print(training_set.class_indices)

#The result contain dimension of batch, so we take the first dimention '0' and only output from the batch '0'
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)