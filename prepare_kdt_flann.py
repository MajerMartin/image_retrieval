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
storage_dir = '/Users/martin.majer/PycharmProjects/PR4/data/kdt_flann_server/'

n = 10000

# <codecell>

with h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:n])
    print 'features.shape:', features.shape

# <codecell>

kdt = img_search.kdtree_flann.ImageSearchKDTreeFlann(storage_dir, 1000000000, (150,150,3))

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    last = 0
    for i in range(1000,n+2,1000):
        img = np.array(fr_imgs['imgs'][last:i][:,:,::-1])
        img = img.astype(np.float32) * 255
        img = img.astype(np.uint8)
        kdt.add_images(img, features[last:i], build_tree=False)
        #for j in range(0, 10):
        #    kdt.add_images(img, features[last:i])
        last = i
        print '-' * 35, i

# <codecell>

kdt.build_tree()

# <codecell>

kdt.save()

# <codecell>

index = 1112
k = 5
neighbors, distances = kdt.find_k_nearest_by_index(index, k)
print 'neighbors:', neighbors
print 'distances:', distances

# <codecell>

images = kdt.get_images(neighbors)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

