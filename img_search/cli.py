__author__ = 'martin.majer'

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from img_search import images
import numpy as np
import caffe
import h5py
import cv2
import time

# image parameters
height = 227.
width = 227.

# root directory of images
root = os.path.join(os.path.dirname(__file__), '..', 'data', 'sun_full', 'SUN397')

# caffe mode toggle
gpu = False

# paths for caffe
caffe_root = '/storage/plzen1/home/campr/metacentrum/caffe/caffe_cemi_7_dev'
caffe_mean_fn = os.path.join(caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
caffe_deploy_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/deploy.prototxt')
caffe_model_fn = os.path.join(caffe_root, 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
caffe_labels_fn = os.path.join(caffe_root, 'data/ilsvrc12/synset_words.txt')

# initialize caffe neural network
net = caffe.Classifier(caffe_deploy_fn,
                       caffe_model_fn,
                       mean=np.load(caffe_mean_fn),
                       # imagenet uses BGR, same as opencv, swap is not needed:
                       #channel_swap=(2,1,0),
                       # bvlc_reference_caffenet uses pixel values [0, 255], but we don't normalize loaded image to [0.0-1.0], so it is scaled to [0.0, 255.0] already after conversion to float32
                       #raw_scale=255,
                       )

# caffe mode
if gpu:
    net.set_mode_gpu()
else:
    net.set_mode_cpu()

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

# paths to all image files
imgs_paths = list_filepaths(root)

i = 0

for path in imgs_paths:
    print '\r', i, path,

    try:
        # open image using opencv
        img = cv2.imread(path)

        # channels
        if len(img.shape) == 3:
            # resize image and crop it (BGR, float32)
            img_cropped = images.resize_crop(img, height, width)
            img_cropped = img_cropped.astype(np.float32)

            # calculate features
            score = net.predict([img_cropped]).flatten()

            i += 1
    except:
        print '\nNot an image'

    if i > 100:
        break
