# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# * zprovoznit h5py knihovnu

# <codecell>

import h5py

# <markdowncell>

# nacitani obrazku, jejich zpracovani
# ===================================
# 
# * v data adresari vytvorit podadresar week01, do nej nakopirovat cca 100 obrazku
# * vytvorit funkci load_imgs(img_dir, shape), ktera nacte vsechny obrazky v adresari a zmeni jejich velikost na (800, 600)
#   * os.listdir
#   * cv2.resize, bicubic interpolation
#   * navratova hodnota dict
#     * {'img1.png': img, ...}  
# * vypocitat prumerny obrazek a zobrazit ho
#   * np.mean

# <codecell>

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

# <codecell>

def load_imgs(img_dir, shape):
    imgs = [img for img in os.listdir(img_dir) if not img.startswith('.')]
    
    resized = {}
    
    for img_name in imgs:
        img_path = img_dir + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_resized = cv2.resize(img, shape, interpolation = cv2.INTER_CUBIC)
        
        resized[img_name] = img_resized
    
    return resized

# <codecell>

width = 800
height = 600

img_dir = '../data/week01ukoly/'
shape = width, height

resized = load_imgs(img_dir,shape)

# <codecell>

img_average = np.zeros((height, width, 3), np.float)

for value in resized.values():
    img_average += value

img_average /= len(resized)
img_average = img_average.astype(np.uint8)

plt.figure()
plt.imshow(img_average)
plt.show()

