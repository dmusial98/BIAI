# import PIL
# print('Pillow Version:', PIL.__version__)
# print("\n")
# print("\n")
# from PIL import Image
# from matplotlib import pyplot

# # load the image
# image = Image.open('idenprof-jpg/idenprof/test/firefighter/firefighter-40.jpg')
# print("photo opened :)")
# print(image.format)
# print(image.mode)
# print(image.size)
# # show the image
# image.show()


# # load and display an image with Matplotlib
# from matplotlib import image
# from matplotlib import pyplot
# # load image as pixel array
# data = image.imread('idenprof-jpg/idenprof/test/firefighter/firefighter-40.jpg')
# # summarize shape of the pixel array
# print(data.dtype)
# print(data.shape)
# # display the array of pixels as an image
# pyplot.imshow(data)
# pyplot.show()


from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from tensorflow.keras.layers import Conv2D
from keras import models
from keras import optimizers

# datagen = ImageDataGenerator()
# # load and iterate training dataset
# train_dir = datagen.flow_from_directory('idenprof-jpg/idenprof/train', class_mode='categorical', batch_size=64)
# # load and iterate test dataset
# test_dir = datagen.flow_from_directory('idenprof-jpg/idenprof/test', class_mode='categorical', batch_size=64)
# # confirm the iterator works
# batchX, batchy = train_dir.next()
# print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

train_dir = 'idenprof-jpg/idenprof/train'
test_dir = 'idenprof-jpg/idenprof/test'
validation_dir = 'idenprof-jpg/idenprof/validation'

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