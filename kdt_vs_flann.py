# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from pyflann import *
import itertools

import img_search

%matplotlib inline

# <codecell>

storage_dir = '/Users/martin.majer/PycharmProjects/PR4/data/kdt/'
n = 10000

# <codecell>

kdt = img_search.kdtree.ImageSearchKDTree(storage_dir, n, (150,150,3))

# <codecell>

index = 3222
k = 5

# <codecell>

# 100 loops, best of 3: 15.3 ms per loop
neighbors = kdt.find_k_nearest_by_index(index, k)

# <codecell>

print 'indexes:\t', neighbors[0]
print 'distances:\t', neighbors[1]

# <codecell>

flann = FLANN()
set_distance_type('euclidean', order = 0)

# <codecell>

params = flann.build_index(np.array(kdt.features), algorithm='kdtree')
print params

# <codecell>

# 10000 loops, best of 3: 178 µs per loop
result, dists = flann.nn_index(kdt.features[index], k)

# <codecell>

result = list(itertools.chain.from_iterable(result))
dists = merged = list(itertools.chain.from_iterable(dists))

print 'indexes:\t', result
print 'distances:\t', dists

# <codecell>

images = kdt.get_images(neighbors[0])

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

# <codecell>

images = kdt.get_images(result)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

