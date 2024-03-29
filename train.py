import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
  
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras.models import Sequential
import pathlib 

import pdb

print("TensorFlow version:", tf.__version__)

data_dir = "screens"

# Training split 
train_ds = tf.keras.utils.image_dataset_from_directory( 
    data_dir, 
    validation_split=0.2, 
    subset="training", 
    seed=123, 
    image_size=(180, 180), 
    batch_size=32
)

# Testing or Validation split 
val_ds = tf.keras.utils.image_dataset_from_directory( 
    data_dir, 
    validation_split=0.2, 
    subset="validation", 
    seed=123, 
    image_size=(180,180), 
    batch_size=32
)
 
class_names = train_ds.class_names 
print(class_names)

 
num_classes = len(class_names) 
  
model = Sequential([ 
    layers.Rescaling(1./255, input_shape=(180,180, 3)), 
    layers.Conv2D(16, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Conv2D(32, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Conv2D(64, 3, padding='same', activation='relu'), 
    layers.MaxPooling2D(), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(num_classes) 
])

model.compile(optimizer='adam', 
    loss=tf.keras.losses.SparseCategoricalCrossentropy( 
        from_logits=True), 
    metrics=['accuracy']) 
model.summary()

epochs = 20
history = model.fit( 
  train_ds, 
  validation_data=val_ds, 
  epochs=epochs 
)

model.save("model_apps_2")
