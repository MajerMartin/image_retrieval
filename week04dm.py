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

# <codecell>

with h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:n])
    print 'features.shape:', features.shape

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    imgs = np.array(fr_imgs['imgs'][:n][:,:,::-1])
    imgs = imgs.astype(np.float32) * 255
    imgs = imgs.astype(np.uint8)
    print 'imgs.shape:', imgs.shape

# <codecell>

dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=n, thumbnail_size=(150,150,3))

# <codecell>

dm.add_images(imgs[:n],features[:n])
print 'distance_matrix.shape:', dm.distance_matrix.shape
print 'len(images):', len(dm.images)
print 'len(features):', len(dm.features)

# <codecell>

dm.add_images(imgs[:2],features[:2])
print 'len(images):', len(dm.images)
print 'len(features):', len(dm.features)

# <codecell>

n = 5000
dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=n, thumbnail_size=(150,150,3))

with h5py.File(h5_imgs_fn,'r') as fr_imgs, h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:n])
    last = 0
    for i in range(0,n,200):
        img = np.array(fr_imgs['imgs'][last:i][:,:,::-1])
        img = img.astype(np.float32) * 255
        img = img.astype(np.uint8)
        dm.add_images(img, features[last:i])
        last = i
        print '\r', i, 'images', '|', len(dm.images), '|', last,

# <codecell>

index = 0
neighbors = dm.find_k_nearest_by_index(index)
print 'neighbors:', neighbors

# <codecell>

images = dm.get_images(neighbors)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

# <codecell>

filename = 'test_dm'
dm.save(filename)

# <codecell>

dm_copy = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=10, thumbnail_size=(10,10,3))
print 'distance_matrix.shape:', dm_copy.distance_matrix
print 'len(images):', len(dm_copy.images)
print 'len(features):', len(dm_copy.features)
print 'max_images:', dm_copy.max_images
print 'thumbnail_size:', dm_copy.thumbnail_size
print '-' * 40

dm_copy.load(filename)

print 'distance_matrix.shape:', dm_copy.distance_matrix.shape
print 'len(images):', len(dm_copy.images)
print 'len(features):', len(dm_copy.features)
print 'max_images:', dm_copy.max_images
print 'thumbnail_size:', dm_copy.thumbnail_size

# <codecell>

(np.triu(dm.distance_matrix) == dm_copy.distance_matrix).all()

