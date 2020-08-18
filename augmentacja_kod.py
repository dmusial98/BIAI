from keras.preprocessing import image

train_judge_dir = "idenprof-jpg/idenprof/train/judge"

import os
fnames = [os.path.join(train_judge_dir, fname) for fname in os.listdir(train_judge_dir)]

img_path = fnames[3]
img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x= x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break

plt.show()    