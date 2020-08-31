from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

conv_base.summary()

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_dir = '../idenprof-jpg/idenprof/train'
validation_dir = '../idenprof-jpg/idenprof/validation'
test_dir = '../idenprof-jpg/idenprof/test'

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

print('Liczba wag poddawanych trenowaniu '
        'przed zamrozeniem bazy: ', len(model.trainable_weights))

#zamrazanie warstw bazy
conv_base._trainable = False

print('Liczba wag poddawanych trenowaniu '
        'po zamrozeniu bazy ', len(model.trainable_weights))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    batch_size=64,
    class_mode='categorical')

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

#trenowanie
history = model.fit_generator(
    train_generator,
    steps_per_epoch=7500/64,
    epochs=30, 
    validation_data=validation_generator,
    validation_steps=1500/64,
    verbose=2)

#wykresy
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
 
plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokaldnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()

plt.show()

#Dostrajanie modelu
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-5),
                metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=7500/64,
    epochs=100, 
    validation_data=validation_generator,
    validation_steps=1500/64)


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point *(1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


plt.plot(epochs, 
        smooth_curve(acc), 'bo', label='Wygladzona dokladnosc trenowania')
plt.plot(epochs,
        smooth_curve(val_acc), 'b', label='Wygladzona dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.legend()

plt.figure()

plt.plot(epochs,
        smooth_curve(loss), 'bo', label='ygladzona strata trenowania')
plt.plot(epochs,
        smooth_curve(val_loss), 'b', label='Wygladzona strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.legend()

plt.show()

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224),
    batch_size=64,
    class_mode='categorical')
