# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# dataset 01
# ==========
# 
# stahnout z http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
# 
# vybrat nahodne 10000 obrazku (seznam vsech souboru glob.glob('*/*') nebo os.listdir(), ten seradit sorted(), np.random.seed(1234), np.random.shuffle(filenames)), ignorovat jednokanalove obrazky
# ty zmensit na 227x227 px, "inteligentne" tak, aby byl zachovan pomer stran a byly oriznute jen okraje
# tyto obrazky ulozit do h5 souboru 
# 
# import h5py
# 
# with h5py.File(fn, 'w') as fr:
#     # n = pocet obrazku
#     fw.create_dataset('imgs', (n, 227, 227, 3), dtype=np.float32)
#     
#     for i, fn in enumerate(filenames):
#         # img = nacist obrazek, zmensit...
#         fw[i] = img
# 
#     fw['filenames'] = filenames

# <codecell>

print 10000*227*227*3*4 / 1024 / 1024

# <codecell>

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import h5py

%matplotlib inline

# <codecell>

def list_filepaths(root):
    imgs_path = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if not name.startswith('.'):
                imgs_path.append(os.path.join(path, name))
                
    return imgs_path

# <codecell>

def keep_ratio(shape, height, width):
    
    if shape[0] <= shape[1]: #height <= width
        ratio = height / shape[0]
        width = shape[1] * ratio
    else:
        ratio = width / shape[1]
        height = shape[0] * ratio
    
    dim = int(width), int(height) #opencv dimension format
    
    return dim

# <codecell>

def crop(img ,height, width):
    if (img.shape[0] == height) and (img.shape[1] == width):
        print 'square'
        return img
    elif img.shape[0] == height:
        print 'horizontal'
        middle = img.shape[1] / 2
        return img[:,(middle - width / 2):(middle + width / 2)]
    else:
        print 'vertical'
        middle = img.shape[0] / 2
        return img[(middle - height / 2):(middle + height / 2),:]

# <codecell>

root = '../data/SUN2012/Images/'

imgs_path = list_filepaths(root)

sample = sorted(imgs_path)  #copy
np.random.seed(50)
np.random.shuffle(sample) #in-place
sample = sample[:10000]

height = 227.
width = 227.

# <codecell>

filename = 'sun_sample.hdf5'

with h5py.File(filename,'w') as f:

# <codecell>

img1 = cv2.imread('../data/1.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../data/2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3 = cv2.imread('../data/2.jpg')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4 = cv2.imread('../data/4.png')
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

abc = img4

dim = keep_ratio(abc.shape, height, width)
img_resized = cv2.resize(abc, dim, interpolation = cv2.INTER_CUBIC)
img_cropped = crop(img_resized, height, width)
print img_resized.shape
print img_cropped.shape

plt.figure()
plt.imshow(img_cropped)
plt.show()






ok = [img1, img2, img3]

for img in ok:
    if len(img.shape) == 3:
        print 'ok'



#for img_path in sample[:2]:
    

