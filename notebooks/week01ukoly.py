# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# zprovoznit h5py knihovnu

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

