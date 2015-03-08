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

kdt = img_search.kdtree.ImageSearchKDTree(max_images=n, thumbnail_size=(150,150,3))

# <codecell>

kdt.add_images(imgs[:n],features[:n])
print 'tree:', kdt.tree
print 'len(images):', len(kdt.images)
print 'len(features):', len(kdt.features)

# <codecell>

kdt.add_images(imgs[:2],features[:2])
print 'len(imgages):', len(kdt.images)
print 'len(features):', len(kdt.features)

# <codecell>

index = 0
neighbors = kdt.find_k_nearest_by_index(index)
print 'neighbors:', neighbors

# <codecell>

images = kdt.get_images(neighbors)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

# <codecell>

filename = 'test_kdt'
kdt.save(filename)

# <codecell>

kdt_copy = img_search.kdtree.ImageSearchKDTree(max_images=10, thumbnail_size=(10,10,3))
print 'tree:', kdt.tree
print 'len(images):', len(kdt_copy.images)
print 'len(features):', len(kdt_copy.features)
print 'max_images:', kdt_copy.max_images
print 'thumbnail_size:', kdt_copy.thumbnail_size
print '-' * 40

kdt_copy.load(filename)

print 'tree:', kdt_copy.tree
print 'len(images):', len(kdt_copy.images)
print 'len(features):', len(kdt_copy.features)
print 'max_images:', kdt_copy.max_images
print 'thumbnail_size:', kdt_copy.thumbnail_size

