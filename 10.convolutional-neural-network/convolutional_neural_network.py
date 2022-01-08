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


