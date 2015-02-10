# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
import cv2

%matplotlib inline

# <codecell>

img_fn = '../data/sun.jpg'
img = cv2.imread(img_fn)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img[:,:,::-1]

print img.shape

# <codecell>

plt.figure()
plt.imshow(img)
plt.show()

# <codecell>

plt.figure()
plt.imshow(img[:,:,0])
plt.colorbar()
plt.gray()
plt.show()

# <codecell>

print img.dtype

