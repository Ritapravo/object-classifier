# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 21:55:44 2019

@author: DELL
"""

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import glob

model=load_model("model2.h5")
classes = ["airplane","car","cat","dog","flower","fruit","motorbike","person"]
def load_image(img_path, show=True):
    img_original = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(64, 64))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_original)
        plt.axis('off')
        plt.show()
    return img_tensor

new_image = load_image(r"H:\natural_object_classifier\sample_images\car2.jpg")                        # load the path of the image as example "E:\89" in place of image path
pred = model.predict(new_image)
idx = np.argmax(np.array(pred[0]))
print(classes[idx])
