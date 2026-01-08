#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1️⃣ Install & Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import json


# In[5]:


# 2️⃣ Dataset Path & Parameters
data_dir = r"C:\Users\Kaviya\OneDrive\Desktop\dataset\PlantVillage Dataset (Labeled)\Color Images"   # change dataset path as needed
img_size = (224,224)
batch_size = 32 
epochs = 10  # increase if GPU available


# In[6]:


# 3️⃣ Data Generator with Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# In[7]:


# 4️⃣ MobileNetV2 Base Model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # freeze base layers


# In[8]:


# 5️⃣ Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)


# In[9]:


# 6️⃣ Compile Model
model.compile(optimizer=Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 7️⃣ Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# 8️⃣ Save Model
model.save("crop_disease_mobilenet.h5")

# 9️⃣ Save Class Names
class_names = list(train_data.class_indices.keys())
with open("class_names.json","w") as f:
    json.dump(class_names,f)

print("✅ MobileNetV2 Model & Class Names Saved Successfully!")


# In[10]:


get_ipython().system('jupyter nbconvert --to python app.ipynb')


# In[11]:


import streamlit as st

st.title("Crop Disease Detection")


# In[ ]:




