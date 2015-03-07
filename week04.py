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
n = 500

dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=n, thumbnail_size=(150,150,3))

# <codecell>

with h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:n])
    print 'features:', features.shape

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    imgs = np.array(fr_imgs['imgs'][:n][:,:,::-1])
    imgs = imgs.astype(np.float32) * 255
    imgs = imgs.astype(np.uint8)
    print 'imgs:', imgs.shape

# <codecell>

dm.add_images(imgs[:n],features[:n])
print dm.distance_matrix.shape
print len(dm.images)
print len(dm.features)

# <codecell>

dm.add_images(imgs[:2],features[:2])

# <codecell>

neighbors = dm.find_k_nearest_by_index(2)
print neighbors

# <codecell>

images = dm.get_images(neighbors)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

# <codecell>

filename = 'test'
dm.save(filename)

# <codecell>

dm_copy = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=10, thumbnail_size=(10,10,3))
dm_copy.load(filename)

# <codecell>

print dm_copy.distance_matrix.shape
print len(dm_copy.images)
print len(dm_copy.features)
print dm_copy.max_images
print dm_copy.thumbnail_size

# <codecell>

(np.triu(dm.distance_matrix) == dm_copy.distance_matrix).all()

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

