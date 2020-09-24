# -*- coding: utf-8 -*-

from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten
from keras.models import Sequential

classifier = Sequential()

classifier.add(Conv2D(32, (6, 6), input_shape=(128, 128, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Conv2D(16, (4, 4), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Conv2D(8, (4, 4), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(3, 3)))

classifier.add(Flatten())

classifier.add(Dense(units=32, activation="relu"))
classifier.add(Dense(units=32, activation="relu"))
classifier.add(Dense(units=16, activation="relu"))
classifier.add(Dense(units=6, activation="softmax"))

classifier.compile(optimizer="adam", loss="categorical_crossentropy")

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('kaggle_bee_vs_wasp',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 11421,
                         epochs = 3,
                         workers=16,
                         max_queue_size=10)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('test//bee_or_wasp3.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][1] == max(result[0]) or result[0][0] == max(result[0]):
    prediction = 'bee'
elif result[0][2] == max(result[0]) or result[0][3] == max(result[0]):
    prediction = 'none'
elif result[0][4] == max(result[0]) or result[0][5] == max(result[0]):
    prediction = 'wasp'

print(prediction)