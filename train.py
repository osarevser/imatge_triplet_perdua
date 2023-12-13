# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:46:25 2023

@author: Oscar
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
import keras
from pathlib import Path
from keras import applications
from keras import layers
from keras import losses
from keras import ops
from keras import optimizers
from keras import metrics
from keras import Model
from keras.applications import resnet
from preprocessat import PreProcessat
from tripletloss import DistanceLayer,SiameseModel

image_count=100

split1_dataset=tf.data.Dataset.from_tensor_slices(np.random.rand(image_count,64,64,3))
split2_dataset=tf.data.Dataset.from_tensor_slices(np.random.rand(image_count,64,64,3))
split3_dataset=tf.data.Dataset.from_tensor_slices(np.random.rand(image_count,64,64,3))

dataset=tf.data.Dataset.zip((split1_dataset,split2_dataset,split3_dataset))

train_dataset=dataset.take(round(image_count*0.8))
val_dataset=dataset.skip(round(image_count*0.8))

train_dataset=train_dataset.batch(32,drop_remainder=False)
val_dataset=val_dataset.batch(32,drop_remainder=False)

target_shape=(64,64)

embedding=keras.Sequential([layers.Input(shape=target_shape+(3,)),
                            layers.Conv2D(32,(7,7),activation='relu',padding='SAME'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(64,(5,5),activation='relu',padding='SAME'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(128,(3,3),activation='relu',padding='SAME'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(256,(1,1),activation='relu',padding='SAME'),
                            layers.MaxPooling2D((2,2)),
                            layers.Conv2D(28,(1,1),activation='relu',padding='SAME'),
                            layers.MaxPooling2D((2,2)),
                            layers.Flatten(),
                            layers.Dense(64)],
                           name="Embedding")

anchor_input=layers.Input(name="anchor",shape=target_shape+(3,))
positive_input=layers.Input(name="positive",shape=target_shape+(3,))
negative_input=layers.Input(name="negative",shape=target_shape+(3,))

distances=DistanceLayer()(embedding(anchor_input),
                          embedding(positive_input),
                          embedding(negative_input))

siamese_network=Model(inputs=[anchor_input,positive_input,negative_input],outputs=distances)

siamese_model=SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
history=siamese_model.fit(train_dataset,epochs=25,validation_data=val_dataset)

#%%
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(history.history["loss"],label="training loss")
plt.plot(history.history["val_loss"],label="validation loss")
plt.legend()