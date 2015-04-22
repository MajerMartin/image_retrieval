__author__ = 'martin.majer'

import os
import cv2
import numpy as np
import h5py

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

root = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sun_full', 'SUN397')

imgs_paths = list_filepaths(root)

print 'Pocet adres:', len(imgs_paths)

i = 0

for path in imgs_paths:
    img = cv2.imread(path)

    if len(img.shape) == 3:
        i += 1

print 'Pocet obrazku:', i