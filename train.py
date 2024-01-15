# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:46:25 2023

@author: Oscar
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
import configparser
import pickle
from keras import layers
from keras import optimizers
from keras import Model
from tripletloss import DistanceLayer,SiameseModel
from image_processing import preprocessing
from embedding import create_embedding
from keras.callbacks import EarlyStopping

config=configparser.ConfigParser()
config.read("config.ini")

version=int(config.get("Version","version"))
path_to_npz=config.get("Paths","path_to_npz")
train_val_ratio=float(config.get("Hyperparameters","train_val_ratio"))
test_images_per_class=int(config.get("Hyperparameters","test_images_per_class"))
train_val_triplets_per_class=int(config.get("Hyperparameters","train_val_triplets_per_class"))
seed=int(config.get("Hyperparameters","seed"))
architecture=config.get("Hyperparameters","architecture")

data=preprocessing(path_to_npz=path_to_npz,
                   train_val_ratio=train_val_ratio,
                   test_images_per_class=test_images_per_class,
                   train_val_triplets_per_class=train_val_triplets_per_class,
                   seed=seed)
train_dataset=data.train_triplets
validation_dataset=data.validation_triplets
test_dataset=data.test_dataset
np.save("versions/v"+str(version)+"_test_dataset.npy",test_dataset)
image_shape=(64,64,1)

print("Summary:")
print("N of classes:                  "+str(len(test_dataset)))
print("Training triplets per class:   "+str(round(train_val_triplets_per_class*train_val_ratio)))
print("Validation triplets per class: "+str(round(train_val_triplets_per_class*(1-train_val_ratio))))
print("Total training triplets:       "+str(train_dataset[0].shape[0]))
print("Total validation triplets:     "+str(validation_dataset[0].shape[0]))

embedding=create_embedding(image_shape=image_shape,len_test_dataset=len(test_dataset),architecture=architecture)

anchor_input=layers.Input(name="anchor",shape=image_shape)
positive_input=layers.Input(name="positive",shape=image_shape)
negative_input=layers.Input(name="negative",shape=image_shape)

distances=DistanceLayer()(embedding(anchor_input),
                          embedding(positive_input),
                          embedding(negative_input))

siamese_network=Model(inputs=[anchor_input,positive_input,negative_input],outputs=distances)

siamese_model=SiameseModel(siamese_network)
siamese_model.compile(optimizer=optimizers.Adam(0.0001))
early_stopping=EarlyStopping(monitor="val_loss",
                             min_delta=0.005,
                             patience=3,
                             mode="min",
                             start_from_epoch=5,
                             restore_best_weights=True)
history=siamese_model.fit(train_dataset,
                          epochs=2,
                          validation_data=validation_dataset,
                          callbacks=[early_stopping])

with open("versions/v"+str(version)+"_training_history.pkl","wb") as history_file:
    pickle.dump(history.history,history_file)
    history_file.close()

plt.figure(figsize=(6,4.5))
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.plot(history.history["loss"],"r-+",label="training loss")
plt.plot(history.history["val_loss"],"b-+",label="validation loss")
plt.legend()
plt.savefig("versions/v"+str(version)+"_loss.png",dpi=400)

embedding.save("versions/v"+str(version)+"_embedding.keras")

with open("versions/v"+str(version)+"_config.ini","w") as configfile:
    config.write(configfile)
    configfile.close()

with open("versions/training_log.txt","a") as training_log:
    training_log.write("Version:                       "+str(version)+"\n")
    training_log.write("Dataset:                       "+path_to_npz+"\n")
    training_log.write("Architecture:                  "+str(architecture)+"\n")
    training_log.write("N of epochs:                   "+str(len(history.history["loss"]))+"\n")
    training_log.write("Training triplets per class:   "+str(round(train_val_triplets_per_class*train_val_ratio))+"\n")
    training_log.write("Validation triplets per class: "+str(round(train_val_triplets_per_class*(1-train_val_ratio)))+"\n")
    training_log.write("Seed:                          "+str(seed)+"\n")
    training_log.write("\n")
    training_log.close()

config.set("Version","version",str(version+1))

with open("config.ini","w") as configfile:
    config.write(configfile)
    configfile.close()