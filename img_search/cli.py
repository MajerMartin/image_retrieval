__author__ = 'martin.majer'

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from img_search import images, kdtree
import numpy as np
import caffe
import cv2

debug = True

# image parameters
height = 227.
width = 227.

# root directory of images
root = '/storage/plzen1/home/mmajer/pr4/data/sun_full/SUN397'

# data directory
if debug:
    storage_dir = os.path.join(root, '..', '..', 'sun_full_sample')
else:
    storage_dir = '/storage/plzen1/home/mmajer/pr4/data/image_search/'

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
print 'Acquiring image paths...'
imgs_paths = list_filepaths(root)
print 'Image paths acquired.'

# sample for testing
if debug:
    sample = sorted(imgs_paths)  #copy
    np.random.seed(50)
    np.random.shuffle(sample) #in-place
    n = 5000
    imgs_paths = sample[:n]

# initialize KDTree
kdt = kdtree.ImageSearchKDTree(storage_dir, 1000000000, (150, 150, 3))

i = 0

for path in imgs_paths:
    print '\r', i, path,

    try:
        # open image using opencv
        img = cv2.imread(path)

        # color channels
        if len(img.shape) == 3:
            # resize image and crop it (BGR, float32)
            img_cropped = images.resize_crop(img, height, width)
            img_cropped = img_cropped.astype(np.float32)

            # calculate features
            score = net.predict([img_cropped]).flatten()

            # convert image to RGB uint8
            img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.uint8)

            # add image and save thumbnail
            kdt.add_images([img_rgb], [score], build_tree=False)

            i += 1
    except:
        print '\nNot an image'

# build KDTree
kdt.build_tree()
kdt.save()

print '\nSample size:', n
print 'Image count:', i
print 'Images added:', len(kdt.features)