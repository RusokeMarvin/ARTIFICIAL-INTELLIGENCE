#!/usr/bin/env python
# coding: utf-8

# In[4]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[5]:


print(os.listdir('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)'))


# In[6]:


pip install tensorflow


# In[7]:


import tensorflow as tf
import tensorflow.keras as keras


# In[11]:


pip install cv


# In[18]:


import numpy as np
import pandas as pd
from keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense,BatchNormalization, Flatten, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Conv2D, Reshape
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0,
    width_shift_range=0.0,
    height_shift_range=0.0,
    brightness_range=None,
    shear_range=0.0,
    zoom_range=0.0,
    channel_shift_range=0.0,
    fill_mode='nearest',
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=None,
    validation_split=0.0,
    interpolation_order=1,
    dtype=None
)


# In[22]:


import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2

#import albumentations as A
#from albumentations.pytorch import ToTensorV2

#import torch
#from torch import nn
#from torch.utils.data import Dataset, DataLoader
#from torchvision.transforms import ToTensor
#from torchvision import models

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[39]:


real = "C:/Users/HP Elitebook Core-i5/OneDrive\Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/"
fake = "C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_fake/"

real_path = os.listdir(real)
fake_path = os.listdir(fake)


# In[40]:


def load_img(path):
    '''Loading images from directory 
    and changing color space from cv2 standard BGR to RGB 
    for better visualization'''
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


# In[43]:


fig = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(real + real_path[i]))
    plt.suptitle("Real faces", fontsize=20)
    plt.axis('off')


# In[ ]:


# FAKE FACES


# In[42]:


fig = plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(load_img(fake + fake_path[i]))
    plt.suptitle("Fake Faces", fontsize=20)
    plt.axis('off')


# In[32]:


from keras.preprocessing.image import ImageDataGenerator


# In[33]:


import cv2


# In[36]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


img = cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00007.jpg')


# In[38]:


image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[39]:


plt.imshow(image)


# In[40]:


plt.imshow(img)


# In[ ]:


# Histogram Equalization


# In[42]:


child= cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00034.jpg',0)


# In[43]:


def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap)
display(child,cmap='gray')


# In[52]:


childd = cv2.equalizeHist(child)


# In[53]:


display(childd,cmap='gray')


# In[ ]:


# COLOR IMAGE


# In[44]:


color_ph = cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00002.jpg')
show_ph = cv2.cvtColor(color_ph,cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(color_ph, cv2.COLOR_BGR2HSV)
display(show_ph)


# In[45]:


hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])


# In[46]:


eq_color_woman = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
display(eq_color_woman)


# In[ ]:


# COLORSPACES


# In[47]:


img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.imshow(img)


# In[49]:


def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


# In[50]:


gamma = 1/4
effected_image = np.power(img, gamma)
display_img(effected_image)


# In[54]:


gamma = 2
effected_image = np.power(childd, gamma)
display_img(effected_image)


# In[55]:


blonde= cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00003.jpg')
blondee = cv2.cvtColor(blonde, cv2.COLOR_BGR2RGB)
display(blondee)


# In[58]:


hat= cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_fake/hard_100_1111.jpg')
hatt = cv2.cvtColor(hat, cv2.COLOR_BGR2RGB)
display(hatt)


# In[ ]:


# Low Pass Filter with a 2D Convolution


# In[59]:


font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(blondee,text='laugh',org=(10,600), fontFace=font,fontScale= 10,color=(0,255,0),thickness=4)
display_img(blondee)


# In[60]:


kernel = np.ones(shape=(5,5),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
display_img(dst)


# In[ ]:


# AVERAGING


# In[61]:


font = cv2.FONT_HERSHEY_COMPLEX
display_img(hatt)


# In[62]:


blurred_img = cv2.blur(hatt,ksize=(5,5))
display_img(blurred_img)


# In[ ]:


Gaussian Blurring


# In[63]:


font = cv2.FONT_HERSHEY_COMPLEX
display_img(blondee)


# In[ ]:


# Median Blurring


# In[65]:


pho = cv2.imread('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00008.jpg')
pho = cv2.cvtColor(pho, cv2.COLOR_BGR2RGB)
display_img(pho)


# In[ ]:


# Bilateral Filtering


# In[66]:


font = cv2.FONT_HERSHEY_COMPLEX
display_img(pho)


# In[ ]:


# Canny Edge Detection


# In[69]:


med_val = np.median(pho) 
lower = int(max(0, 0.7* med_val))


# In[70]:


upper = int(min(255,1.3 * med_val))


# In[71]:


blurred_img = cv2.blur(pho,ksize=(5,5))


# In[72]:


edges = cv2.Canny(image=blurred_img, threshold1=lower , threshold2=upper)


# In[73]:


plt.imshow(edges)


# In[95]:


import pandas as pd


# In[96]:


pip install split-folders


# In[57]:


_img = keras.preprocessing.image.load_img('C:/Users/HP Elitebook Core-i5/OneDrive/Desktop/archive (1)/real_and_fake_face_detection/real_and_fake_face/training_real/real_00009.jpg' ,target_size=(224,224))
plt.imshow(_img)
plt.show()


# In[58]:


import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
print('tensorflow {}'.format(tf.__version__))
print("keras {}".format(keras.__version__))
import matplotlib.pyplot as plt
model = keras.applications.VGG16(weights='imagenet')


# In[59]:


#preprocess image to get it into the right format for the model
img = keras.preprocessing.image.img_to_array(_img)
img = img.reshape((1, *img.shape))
y_pred = model.predict(img)


# In[60]:


layers = [layer.output for layer in model.layers]


# In[61]:


images = tf.Variable(img, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]
    
grads = tape.gradient(loss, images)


# In[62]:


grads.shape


# In[63]:


dgrad_abs = tf.math.abs(grads)


# In[64]:


dgrad_max_ = np.max(dgrad_abs, axis=3)[0]


# In[65]:


dgrad_max_.shape


# In[66]:


arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)


# In[67]:


fig, axes = plt.subplots(1,2,figsize=(14,5))
axes[0].imshow(_img)
i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
fig.colorbar(i)


# In[ ]:




