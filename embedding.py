# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 18:12:34 2023

@author: Oscar
"""
import keras
from keras import layers

def create_embedding(image_shape,len_test_dataset,architecture):
    keras.backend.clear_session()
    if architecture=="a1":
        embedding=keras.Sequential([layers.Input(shape=image_shape),
                                    layers.Conv2D(int(0.5*len_test_dataset),(7,7),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(1*len_test_dataset,(5,5),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(4*len_test_dataset,(1,1),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(1,1),activation=None,padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Flatten(),
                                    layers.Dense(len_test_dataset,activation=None)],
                                   name="a1")
    elif architecture=="a2":
        embedding=keras.Sequential([layers.Input(shape=image_shape),
                                    layers.Conv2D(int(0.5*len_test_dataset),(7,7),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(1*len_test_dataset,(5,5),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(4*len_test_dataset,(1,1),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(1,1),activation=None,padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Flatten(),
                                    layers.Dense(2*len_test_dataset,activation="relu"),
                                    layers.Dense(1*len_test_dataset,activation=None)],
                                   name="a2")
    elif architecture=="a3":
        embedding=keras.Sequential([layers.Input(shape=image_shape),
                                    layers.Conv2D(int(0.5*len_test_dataset),(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(1*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(4*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(8*len_test_dataset,(3,3),activation=None,padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Flatten(),
                                    layers.Dense(4*len_test_dataset,activation="relu"),
                                    layers.Dense(2*len_test_dataset,activation="relu"),
                                    layers.Dense(1*len_test_dataset,activation=None)],
                                   name="a3")
    elif architecture=="a4":
        embedding=keras.Sequential([layers.Input(shape=image_shape),
                                    layers.Conv2D(int(0.5*len_test_dataset),(9,9),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(1*len_test_dataset,(7,7),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(2*len_test_dataset,(5,5),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(4*len_test_dataset,(3,3),activation="relu",padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Conv2D(8*len_test_dataset,(1,1),activation=None,padding="SAME"),
                                    layers.MaxPooling2D((2,2)),
                                    layers.Flatten(),
                                    layers.Dense(4*len_test_dataset,activation="relu"),
                                    layers.Dense(2*len_test_dataset,activation="relu"),
                                    layers.Dense(1*len_test_dataset,activation=None)],
                                   name="a4")
    else:
        embedding=None
    return embedding