# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import h5py

import img_search

# <codecell>

h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'

# <codecell>

n = 3000

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

%%timeit
dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=10000, thumbnail_size=(150,150,3))

# <codecell>

%%timeit
kdt = img_search.kdtree.ImageSearchKDTree(max_images=10000, thumbnail_size=(150,150,3))

# <codecell>

%%timeit -n 20
dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=10000, thumbnail_size=(150,150,3))
dm.add_images(imgs[:n],features[:n])

# <codecell>

%%timeit -n 20
kdt = img_search.kdtree.ImageSearchKDTree(max_images=10000, thumbnail_size=(150,150,3))
kdt.add_images(imgs[:n],features[:n])

# <codecell>

dm = img_search.distance_matrix.ImageSearchDistanceMatrix(max_images=10000, thumbnail_size=(150,150,3))
dm.add_images(imgs[:n],features[:n])

kdt = img_search.kdtree.ImageSearchKDTree(max_images=10000, thumbnail_size=(150,150,3))
kdt.add_images(imgs[:n],features[:n])

# <codecell>

index = 0

# <codecell>

%%timeit
dm_neighbors = dm.find_k_nearest_by_index(index)

# <codecell>

%%timeit
kdt_neighbors = kdt.find_k_nearest_by_index(index)

