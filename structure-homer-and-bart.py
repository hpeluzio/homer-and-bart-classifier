import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import keras.utils as image

classifier = Sequential()
classifier.add(Conv2D(32, (3,3), input_shape = (64, 64, 3)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 128, activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',
               metrics = ['accuracy'])

generator_training = ImageDataGenerator(rescale = 1./255, rotation_range = 7,
                                        horizontal_flip = True, shear_range = 0.2,
                                        height_shift_range = 0.07, zoom_range = 0.2)

generator_test= ImageDataGenerator(rescale = 1./255)

base_training = generator_training.flow_from_directory('dataset_characters/training_set', 
                                                       target_size = (64, 64),
                                                       batch_size = 10,
                                                       class_mode = 'binary')

base_test = generator_test.flow_from_directory('dataset_characters/test_set', 
                                                       target_size = (64, 64),
                                                       batch_size = 10,
                                                       class_mode = 'binary')

classifier.fit_generator(base_training, steps_per_epoch = 196 / 1,
                         epochs = 100, validation_data = base_test,
                         validation_steps = 73 / 1)

# base_training.class_indices
base_training.class_indices
#image_test = image.load_img('dataset_characters/test_set/homer/homer1.bmp',  target_size = (64,64))
image_test = image.load_img('dataset_characters/test_set/bart/bart4.bmp',  target_size = (64,64))
image_test = image.img_to_array(image_test)
image_test
image_test /= 255
image_test = np.expand_dims(image_test, axis = 0)
prediction_test = classifier.predict(image_test)
