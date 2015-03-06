# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2

import img_search

%matplotlib inline

# <codecell>

h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'

dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=500, thumbnail_size=(150,150,3))

# <codecell>

with h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:500])
    print 'features:', features.shape

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    imgs = np.array(fr_imgs['imgs'][:500][:,:,::-1])
    imgs = imgs.astype(np.float32) * 255
    print 'imgs:', imgs.shape

# <codecell>

dm.add_images(imgs[:400],features[:400])
print dm.distance_matrix.shape
print len(dm.images)
print len(dm.features)

# <codecell>

class ImageSearchKDTree(object):
     def __init__(self, max_images=100000):
         self.max_images = max_images
         self.distance_matrix = None
         self.images = []
         self.features = []   
            
     def add_images(self, images, features):
          pass
            
     def find_k_nearest_by_index(self, img_index, k=3):
          pass
            
     def get_images(self, indexes):
          pass
            
# zmerit rychlost pridavani obrazk a hledani nejblizsich            

