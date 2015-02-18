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

root = '../data/SUN2012/Images/'

imgs_path = list_filepaths(root)

sample = sorted(imgs_path)  #copy
np.random.seed(50)
np.random.shuffle(sample) #in-place
sample = sample[:10000]

