import os
import zipfile
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Unzip and save the file
local_zip = "C:/Github/TensorFlow/Course_Material/tmp/cats_and_dogs_filtered.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall("C:/Github/TensorFlow/Course_Material/tmp")
zip_ref.close()

#Setup Data set
base_dir = "C:/Github/TensorFlow/Course_Material/tmp/cats_and_dogs_filtered"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#See what the filenames look like
train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

#Total number of cat and dog images
print('total training cat images: ', len(os.listdir(train_cats_dir))) # ctrl + D: to copy the line
print('total training dog images: ', len(os.listdir(train_dogs_dir)))


print('total validation cat images: ', len(os.listdir(validation_cats_dir)))
print('total validation dog images: ', len(os.listdir(validation_dogs_dir)))

'''
# See a few pictures from our Dataset

nrows = 4
ncols = 4
pic_index = 0 #Index for iterating over images

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index += 8

next_cat_pix = [os.path.join(train_cats_dir, fname) for fname in train_cat_fnames[pic_index-8: pic_index]]

next_dog_pix = [os.path.join(train_dogs_dir, fname) for fname in train_dog_fnames[pic_index-8: pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
    
plt.show()
'''

#Build the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation= 'relu', input_shape= (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation= 'relu'),
    tf.keras.layers.Dense(1, activation= 'sigmoid')])

model.summary()

#Compile the Model
model.compile(optimizer = RMSprop(lr=0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#RMSprop automates learning rate tuning for us, Adam and Adagrad do this also...

#Data Preprocessing
train_datagen = ImageDataGenerator(rescale = 1.0/255.)
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

#Flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=20, class_mode='binary', target_size=(150,150))
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size=20, class_mode='binary', target_size=(150,150))

#Train the Model
history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs=15, validation_steps = 50, verbose=2)

'''
#Test the Model using own image

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
    path = 'content' + fn
    img = image.load_img(path, target_size = (150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size = 10)
    
    print(classes[0])
    
    if classes[0] > 0:
        print(fn + "is a dog")
    else:
        print(fn + "is a cat")
'''

'''
# Visualizing Intermediate Representations
import numpy as np
import random
from tensorflow.keras.preprocessing.iamge import img_to_array, load_img

#Define a new Model, take an image as input and output intermediate representations for all layers
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outpus = successive_outputs)

cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]

img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150)) #This is a PIL image

x = img_to_array(img)
x = x.reshape((1,) + x.shape)
x /= 255.0

#Run the image through the model and obtain all intermediate representations
successive_feature_maps = visualization_model.predict(x)
layer_names = [layer.name for layer in model.layers]

#Display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size*n_features))

        #Postprocess the feature to be visually palatable
        for i in range(n_features):
            x = feature_map[0,:,:,i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i*size : (i+1)*size] = x

        #Display the grid
        scale = 20./n_features
        plt.figure(figsize = (scale*n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap = 'viridis')

        #From the raw pixels of the images to increasingly abstract and compact representations
'''

'''
# Evaluating Accuracy and Loss for the Model
acc = history.history['accuracy']
loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

epochs = range(len(acc)) #Number of epochs

#Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

#Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
'''

