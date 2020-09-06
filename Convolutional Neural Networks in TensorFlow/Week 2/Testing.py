from tensorflow import keras
import numpy as np
from keras.preprocessing import image

new_model = keras.models.load_model('my_model.h5')

img = image.load_img("C:/Users/shens/Downloads/Collage_of_Nine_Dogs.jpg", target_size = (150,150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = new_model.predict(images, batch_size=10)
print(classes[0])
if classes[0] > 0.5:
    print("It is a dog")
else:
    print("It is a cat")