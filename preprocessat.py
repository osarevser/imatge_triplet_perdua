"""
Created on Fri Dec  1 16:57:28 2023

@author: Oscar
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PreProcessing:
    def __init__(self,path):
        self.path=path
        print("Loading and processing dataset...")
        self.classes,self.labels_train,self.images_train,self.labels_test,self.images_test=self.preprocessing(0.9)
        #self.map_train_label_indices={label:np.flatnonzero(self.labels_train==label) for label in self.classes}
        print('Preprocessing Done. Summary:')
        print("Images train :", self.images_train.shape)
        print("Labels train :", self.labels_train.shape)
        print("Images test  :", self.images_test.shape)
        print("Labels test  :", self.labels_test.shape)
        print("N of classes :", len(self.classes))

    def read_dataset(self):
        data=np.load(self.path)
        classes,labels,images=[],[],[]
        label=0
        for k,v in data.items():
            classes.append(k)
            for img in v:
                labels.append(label)
                images.append(img)
            label+=1
        return np.asarray(classes),np.asarray(labels),np.asarray(images)
    
    def preprocessing(self,train_test_ratio):
        classes,labels,images=self.read_dataset()
        images=(images-images.min(axis=0))/(images.max(axis=0)-images.min(axis=0))
        shuffle_indices=np.random.permutation(np.arange(len(labels)))
        labels_shuffled,images_shuffled=labels[shuffle_indices],images[shuffle_indices,:,:]
        train_test=int(train_test_ratio*len(labels))
        return classes,labels_shuffled[:train_test],images_shuffled[:train_test],labels_shuffled[train_test:],images_shuffled[train_test:]
    
    def get_triplets(self):
        label_l,label_r=np.random.choice(self.unique_train_label,2,replace=False)
        a,p=np.random.choice(self.map_train_label_indices[label_l],2,replace=False)
        n=np.random.choice(self.map_train_label_indices[label_r])
        return a,p,n
    
    def get_triplets_batch(self,n):
        idxs_a,idxs_p,idxs_n=[],[],[]
        for _ in range(n):
            a,p,n=self.get_triplets()
            idxs_a.append(a)
            idxs_p.append(p)
            idxs_n.append(n)
        return self.images_train[idxs_a,:],self.images_train[idxs_p,:],self.images_train[idxs_n,:]