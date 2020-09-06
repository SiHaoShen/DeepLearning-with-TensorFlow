import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from urllib.request import urlopen
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile

#Download Inception pretrained Model
filedata = urlopen('https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
datatowrite = filedata.read()

with open('C:/Github/TensorFlow/Course_Material/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', 'wb') as f:
    f.write(datatowrite)

local_weights_file = "C:/Github/TensorFlow/Course_Material/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

#Without fully connected top layer, without build in weights
pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

#pre_trained_model.summary()

#Choose last layer from the mixed7 layer
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

#Add Last output layers
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#Download Cats and dogs dataset
filedata = urlopen('https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip')
datatowrite = filedata.read()

with open('C:/Github/TensorFlow/Course_Material/tmp/cats_and_dogs_filtered.zip', 'wb') as f:
    f.write(datatowrite)

#Unzip the Dataset
local_zip = "C:/Github/TensorFlow/Course_Material/tmp/cats_and_dogs_filtered.zip"

zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall("C:/Github/TensorFlow/Course_Material/tmp")
zip_ref.close()

#Define our directories and files
base_dir = "C:/Github/TensorFlow/Course_Material/tmp/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

#Add data augmentation parameters
train_datagen = ImageDataGenerator(rescale = 1.0/255.,
                                   rotation_range = 40,
                                   width_shift_range= 0.2,
                                   height_shift_range= 0.2,
                                   shear_range =0.2,
                                   zoom_range =0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

#Flow training/testing images in batches of 20 using generators
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode= 'binary',
                                                    target_size = (150,150))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    batch_size = 20,
                                                    class_mode= 'binary',
                                                    target_size = (150,150))

history = model.fit(train_generator,
                    validation_data = validation_generator,
                    steps_per_epoch= 100,
                    epochs = 20,
                    validation_steps = 50,
                    verbose = 2)

#Show Accuracy and Loss for each epoch
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label = 'Training accuracy')
plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
