from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.layers import Conv2D
from keras import models
from keras import optimizers

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_dir = '../..idenprof-jpg/idenprof/train'
test_dir = '../..idenprof-jpg/idenprof/test'
validation_dir = '../..idenprof-jpg/idenprof/validation'

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu',
                    input_shape = (224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (224,224),
    batch_size = 64,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (224,224),
    batch_size = 64,
    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (224,224),
    batch_size = 64,
    class_mode = 'categorical')

for data_batch, labels_batch in train_generator:
    print(data_batch.shape)
    print(labels_batch.shape)
    break

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 7500/64,
    epochs = 30,
    validation_data = validation_generator,
    validation_steps = 1500/64)

model.save('idenprof_attempt_1.h5')
