# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:42:35 2023

@author: Oscar
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#sorting=np.random.randint(0,)

data=np.load("pdb104M_noisy.npz")
labels,images,lens=[],[],[]
for k,v in data.items():
    labels.append(k)
    images.append(v)
    lens.append(v.shape[0])

#%%
N_triplets_per_class=160
np.random.seed(185422)
random_key_anchor_list,random_key_positive_list=[],[]
for l in lens:
    random_key_anchor=np.random.randint(0,l,N_triplets_per_class)
    random_key_positive=np.random.randint(0,l,N_triplets_per_class)
    while np.any(random_key_anchor==random_key_positive):
        random_key_positive=(random_key_anchor!=random_key_positive)*random_key_positive+(random_key_anchor==random_key_positive)*np.random.randint(0,l,N_triplets_per_class)
    random_key_anchor_list.append(random_key_anchor)
    random_key_positive_list.append(random_key_positive)

anchor,positive=[],[]
for i in range(64):
    anchor.append(images[i][random_key_anchor_list[i],:])
    positive.append(images[i][random_key_positive_list[i],:])
    
anchor=np.concatenate(anchor)
positive=np.concatenate(positive)
#%%
negative=[]
for i in range(len(labels)):
    negative_i=[]
    for n in range(N_triplets_per_class):
        random_class=i
        while random_class==i:
            random_class=np.random.randint(0,len(labels))
        negative_i.append(images[random_class][np.random.randint(0,lens[random_class]),:])
    negative.append(np.stack(negative_i))
negative=np.concatenate(negative)
#%%
min_val,max_val=np.min((anchor.min(),positive.min(),negative.min())),np.max((anchor.max(),positive.max(),negative.max()))
anchor=(anchor-min_val)/(max_val-min_val)
positive=(positive-min_val)/(max_val-min_val)
negative=(negative-min_val)/(max_val-min_val)
#%%
shuffle=np.random.shuffle(np.arange(len(anchor)))
anchor=anchor[shuffle,:][0,:]
positive=positive[shuffle,:][0,:]
negative=negative[shuffle,:][0,:]
#%%
np.save("anchor.npy",anchor)
np.save("positive.npy",positive)
np.save("negative.npy",negative)