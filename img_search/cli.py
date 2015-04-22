__author__ = 'martin.majer'

import os
import cv2
import numpy as np
import h5py

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from img_search import images

def list_filepaths(root):
    '''
    Get paths to all images.
    :param root: root directory
    :return: list of image paths
    '''
    imgs_paths = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if not name.startswith('.'):
                imgs_paths.append(os.path.join(path, name))

    return imgs_paths

root = os.path.join(os.path.dirname(__file__), '..', 'data', 'sun_full', 'SUN397')

print root

imgs_paths = list_filepaths(root)

i = 0

for path in imgs_paths:
    print '\r', i, path,

    img = cv2.imread(path)

    try:
        if len(img.shape) == 3:
            i += 1
    except:
        print 'Nonetype'

    if i > 5000:
        break

print 'Pocet adres:', len(imgs_paths)
print 'Pocet obrazku:', i