import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
from urllib.request import urlopen
import glob
import sys
#from google.colab import files

#Download Dataset
filedata = urlopen('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip')
datatowrite = filedata.read()

with open('C:/Github/TensorFlow/Course_Material/tmp/rps.zip', 'wb') as f:
    f.write(datatowrite)

local_zip = 'C:/Github/TensorFlow/Course_Material/tmp/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Github/TensorFlow/Course_Material/tmp/')
zip_ref.close()

#Download Test Dataset
filedata = urlopen('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps-test-set.zip')
datatowrite = filedata.read()

with open('C:/Github/TensorFlow/Course_Material/tmp/rps-test-set.zip', 'wb') as f:
    f.write(datatowrite)

local_zip = 'C:/Github/TensorFlow/Course_Material/tmp/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('C:/Github/TensorFlow/Course_Material/tmp/')
zip_ref.close()

#Join Rock, Paper, Scissor Datapath
rock_dir = os.path.join('C:/Github/TensorFlow/Course_Material/tmp/rps/rock')
paper_dir = os.path.join('C:/Github/TensorFlow/Course_Material/tmp/rps/paper')
scissors_dir = os.path.join('C:/Github/TensorFlow/Course_Material/tmp/rps/scissors')

#Show image path and filename
rock_files = os.listdir(rock_dir)
print(rock_files[:10])
paper_files = os.listdir(paper_dir)
print(paper_files[:10])
scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

'''
#Show some images from the dataset
pic_index = 2
next_rock = [os.path.join(rock_dir, fname)
             for fname in rock_files[pic_index-2:pic_index]]

next_paper = [os.path.join(paper_dir, fname)
              for fname in paper_files[pic_index-2:pic_index]]

next_scissors = [os.path.join(scissors_dir, fname)
              for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock + next_paper + next_scissors):
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
'''
#Setup Image Data Generator for training and testing using Dataaugmentation etc.
TRAINING_DIR = "C:/Github/TensorFlow/Course_Material/tmp/rps/"
training_datagen = ImageDataGenerator(
    rescale = 1./255.,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')

VALIDATION_DIR = "C:/Github/TensorFlow/Course_Material/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(150,150), class_mode = 'categorical', batch_size=126)

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR, target_size=(150,150), class_mode = 'categorical', batch_size=126)

#Setup Model structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape= (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')])

#model.summary and Model Training
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
history = model.fit(train_generator, epochs=15, steps_per_epoch=20, validation_data = validation_generator, verbose = 2, validation_steps = 3)
#Save the model
model.save("rps.h5")


'''
#Plot the Model accuracy and loss history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
'''

#Test the model out
#uploaded = files.upload()
#for fn in uploaded.keys():
#    path = fn
new_model = keras.models.load_model('rps.h5')
#Save Output Data
stdoutOrigin = sys.stdout
sys.stdout = open("log.txt", "w")

for file in glob.glob("C:/Users/shens/Downloads/rps-validation/*.png"):
    path = file
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print(file)
    print(classes)  # Paper = (1,0,0), Rock = (0,1,0), Scissor = (0,0,1)
    if classes[0][0] == 1:
        print("It is a Paper")
    else:
        if classes[0][1] == 1:
            print("It is a Rock")
        else:
            print("It is a Scissor")



sys.stdout.close()
sys.stdout=stdoutOrigin

'''
#Select csv Data
    def get_data(file_path):
        with open(file_path) as file:
            data = csv.reader(file)
            labels = []
            images = []
            for i, row in enumerate(data):
                if i == 0:
                    continue
                labels.append(row[0])
                image = row[1:]
                image_array = np.array_split(image, 28)
                images.append(image_array)
            labels = np.array(labels).astype(float)
            images = np.array(images).astype(float)
        return images, labels
'''