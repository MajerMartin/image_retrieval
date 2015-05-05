__author__ = 'martin.majer'

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import h5py
import cv2

root = '/storage/plzen1/home/mmajer/pr4/data/sun_full/SUN397'
storage = '/storage/plzen1/home/mmajer/pr4/data/'
filename = storage + 'sun_img_names'

def list_filepaths(root):
    '''
    Get paths to all images.
    :param root: root directory
    :return: list of image paths
    '''
    imgs_paths = []
    imgs_names = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if not name.startswith('.'):
                imgs_paths.append(os.path.join(path, name))
                imgs_names.append(name)

    return imgs_paths, imgs_names

# paths to all image files
imgs_paths, img_names = list_filepaths(root)
zipped = zip(imgs_paths, img_names)

i = 0

with h5py.File(filename, 'w') as fw:
    fw.create_dataset('filename', (len(imgs_paths), ), dtype=h5py.special_dtype(vlen=unicode))
    fw.create_dataset('path', (len(imgs_paths), ), dtype=h5py.special_dtype(vlen=unicode))

    for path, name in zipped:
        if i % 50 == 0:
            print '\r', i, path,

        try:
            # open image using opencv
            img = cv2.imread(path)

            # color channels
            if len(img.shape) == 3:
                fw['filename'][i] = name
                fw['path'][i] = path
                i += 1
        except:
            print '\nNot an image'