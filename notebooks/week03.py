# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import h5py

%matplotlib inline

# <codecell>

h5_imgs_fn = '../data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'
n = 10000

# <codecell>

with h5py.File(h5_imgs_fn,'r') as fr_imgs, h5py.File(h5_fts_fn,'r') as fr_features:
    features = np.copy(fr_features['score'])
    print features.shape
    dst_matrix = get_distance_matrix(features, n)
    print dst_matrix.shape

# <codecell>

img_index = 1920
k = 1
find_k_nearest(dst_matrix, img_index, k)

# <headingcell level=3>

# Distance Matrix

# <codecell>

def get_distance_matrix(features, n):
    
    dst_matrix = np.zeros((n,n))
    
    for i in range(0, n):
        print '\r', 'i: ', i,
        for j in range(i, n):
            dst_matrix[i,j] = np.linalg.norm(features[i] - features[j])
        
    return np.array(dst_matrix)

# <headingcell level=3>

# Distance Matrix - scipy

# <headingcell level=3>

# K-Nearest Neighbors

# <codecell>

def find_k_nearest(dst_matrix, img_index, k):
    
    dst = dst_matrix[img_index,img_index:]
    best_indexes = np.argsort(dst)
    
    with h5py.File(h5_imgs_fn,'r') as fr_imgs:
    
        for i in range(0, k+1):
           index = best_indexes[i] + img_index
           img = np.array(fr_imgs['imgs'][index][:,:,::-1])
           img = img.astype(np.float32) * 255
        
           plt.figure()
           plt.imshow(img.astype(np.uint8)[:,:,::-1])
           plt.show()

# <headingcell level=3>

# K-Nearest Neighbors - sklearn

