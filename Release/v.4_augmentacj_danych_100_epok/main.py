from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.layers import Conv2D
from keras import models
from keras import optimizers

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_dir = "../..idenprof-jpg/idenprof/train"
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation = 'relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

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

# from keras.preprocessing import image

# train_judge_dir = "idenprof-jpg/idenprof/train/firefighter"

# import os
# fnames = [os.path.join(train_judge_dir, fname) for fname in os.listdir(train_judge_dir)]

# img_path = fnames[3]
# img = image.load_img(img_path, target_size=(224, 224))

# x = image.img_to_array(img)
# x= x.reshape((1,) + x.shape)

# import matplotlib.pyplot as plt
# i = 0
# for batch in train_datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break

# plt.show()    

history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50)


model.save('idenprof_attempt_1.h5')