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
storage_dir = '/Users/martin.majer/PycharmProjects/PR4/data/kdt_server/'

#h5_imgs_fn = '/storage/plzen1/home/mmajer/pr4/data/sun_sample.hdf5'
#h5_fts_fn = h5_imgs_fn + '.features.hdf5'
#storage_dir = '/storage/plzen1/home/mmajer/pr4/data/image_search/'

n = 10000

# <codecell>

with h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'][:n])
    print 'features.shape:', features.shape

# <codecell>

kdt = img_search.kdtree.ImageSearchKDTree(storage_dir, 1000000000, (150,150,3))

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    last = 0
    for i in range(1000,n+2,1000):
        img = np.array(fr_imgs['imgs'][last:i][:,:,::-1])
        img = img.astype(np.float32) * 255
        img = img.astype(np.uint8)
        kdt.add_images(img, features[last:i])
        #for j in range(0, 10):
        #    kdt.add_images(img, features[last:i])
        last = i
        print '-' * 35, i

# <codecell>

#from sklearn.neighbors import KDTree
#%load_ext memory_profiler
#%memit -r 10 KDTree(kdt.features, metric='euclidean')
#%timeit -n 10 KDTree(kdt.features, metric='euclidean')

print ' 20000 features\t|\t10 loops, best of 3: 1.99 s per loop\t|\tpeak memory: 415.19 MiB, increment: 152.59 MiB'
print ' 40000 features\t|\t10 loops, best of 3: 6.45 s per loop\t|\tpeak memory: 626.06 MiB, increment: 321.25 MiB'
print ' 60000 features\t|\t10 loops, best of 3: 10.4 s per loop\t|\tpeak memory: 764.65 MiB, increment: 457.90 MiB'
print ' 80000 features\t|\t10 loops, best of 3: 15.6 s per loop\t|\tpeak memory: 919.38 MiB, increment: 610.48 MiB'
print '100000 features\t|\t10 loops, best of 3: 22.6 s per loop\t|\tpeak memory: 1135.96 MiB, increment: 825.71 MiB'

# <codecell>

kdt.save()

# <codecell>

index = 765
k = 5
neighbors = kdt.find_k_nearest_by_index(index, k)
print 'neighbors:', neighbors

# <codecell>

images = kdt.get_images(neighbors)

for img in images:
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.colorbar()
    plt.show()

