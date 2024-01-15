# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:50:38 2023

@author: Oscar
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class preprocessing:
    def __init__(self,path_to_npz,train_val_ratio,test_images_per_class,train_val_triplets_per_class,seed):
        print("Loading dataset...")
        self.path_to_npz=path_to_npz
        self.train_val_ratio=train_val_ratio
        self.test_images_per_class=test_images_per_class
        self.train_val_triplets_per_class=train_val_triplets_per_class
        self.seed=seed
        np.random.seed(seed)
        self.labels,self.images,self.lens=self.load_data()
        self.train_dataset,self.validation_dataset,self.test_dataset=self.split_data()
        self.test_dataset=self.normalize(self.test_dataset)
        self.train_triplets=self.generate_triplets(image_list=self.train_dataset,
                                                   N_triplets_per_class=round(self.train_val_triplets_per_class*self.train_val_ratio),
                                                   seed=self.seed)
        self.validation_triplets=self.generate_triplets(image_list=self.validation_dataset,
                                                        N_triplets_per_class=round(train_val_triplets_per_class*(1-train_val_ratio)),
                                                        seed=self.seed)
        
    def load_data(self):
        data=np.load(self.path_to_npz)
        labels,images,lens=[],[],[]
        for k,v in data.items():
            labels.append(k)
            images.append(v)
            lens.append(v.shape[0])
        return labels,images,lens
    
    def split_data(self):
        train_data=[]
        validation_data=[]
        test_data=[]
        i=0
        for classs in self.images:
            if i%1==0:
                np.random.shuffle(classs)
                train_len=round((classs.shape[0]-self.test_images_per_class)*self.train_val_ratio)
                train_data.append(classs[:train_len,:])
                validation_data.append(classs[train_len:-self.test_images_per_class])
                test_data.append(classs[-self.test_images_per_class:,:])
            i+=1
        return train_data,validation_data,test_data
    
    def normalize(self,dataset):
        normal_dataset=[]
        max_val,min_val=np.max(dataset),np.min(dataset)
        for classs in dataset:
            classs=(classs-min_val)/(max_val-min_val)
            normal_dataset.append(classs)
        return normal_dataset
    
    def generate_triplets(self,image_list,N_triplets_per_class,seed):
        random_key_anchor_list,random_key_positive_list=[],[]
        for classs in image_list:
            random_key_anchor=np.random.randint(0,classs.shape[0],N_triplets_per_class)
            random_key_positive=np.random.randint(0,classs.shape[0],N_triplets_per_class)
            while np.any(random_key_anchor==random_key_positive):
                random_key_positive=(random_key_anchor!=random_key_positive)*random_key_positive+(random_key_anchor==random_key_positive)*np.random.randint(0,classs.shape[0],N_triplets_per_class)
            random_key_anchor_list.append(random_key_anchor)
            random_key_positive_list.append(random_key_positive)

        anchor,positive=[],[]
        for i in range(len(image_list)):
            anchor.append(image_list[i][random_key_anchor_list[i],:])
            positive.append(image_list[i][random_key_positive_list[i],:])
            
        anchor=np.concatenate(anchor)
        positive=np.concatenate(positive)
        
        negative=[]
        for i in range(len(image_list)):
            negative_i=[]
            for n in range(N_triplets_per_class):
                random_class=i
                while random_class==i:
                    random_class=np.random.randint(0,len(image_list))
                negative_i.append(image_list[random_class][np.random.randint(0,image_list[random_class].shape[0]),:])
            negative.append(np.stack(negative_i))
        negative=np.concatenate(negative)
        
        min_val,max_val=np.min((anchor.min(),positive.min(),negative.min())),np.max((anchor.max(),positive.max(),negative.max()))
        anchor=(anchor-min_val)/(max_val-min_val)
        positive=(positive-min_val)/(max_val-min_val)
        negative=(negative-min_val)/(max_val-min_val)
        
        shuffle=np.arange(len(anchor))
        np.random.shuffle(shuffle)
        anchor_s=anchor[shuffle,:]
        positive_s=positive[shuffle,:]
        negative_s=negative[shuffle,:]
        return anchor_s,positive_s,negative_s
        