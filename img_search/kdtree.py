__author__ = 'martin.majer'

import numpy as np
import cv2
import h5py
import itertools
from sklearn.neighbors import KDTree


class ImageSearchKDTree(object):
    def __init__(self, max_images=100000, thumbnail_size=(150,150,3)):
        self.max_images = max_images
        self.thumbnail_size = thumbnail_size
        self.tree = None
        self.images = []
        self.features = []

    def add_images(self, images, features):
        dim = (self.thumbnail_size[0], self.thumbnail_size[1])

        # add resized images
        if len(images.shape) == 3:
            if (len(self.images) + 1) > self.max_images:
                print 'You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.images))
                return
            else:
                img_resized = cv2.resize(images, dim, interpolation = cv2.INTER_NEAREST)  # INTER_CUBIC changes pixel values
                self.images.append(img_resized)
        else:
            if (len(self.images) + len(images)) > self.max_images:
                print 'You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.images))
                return
            else:
                for img in images:
                    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
                    self.images.append(img_resized)

        # add features
        if len(features.shape) == 1:
            self.features.append(features)
        else:
            self.features.extend(features)

        self.tree = KDTree(self.features, metric='euclidean')

    def find_k_nearest_by_index(self, img_index, k_neighbors=3):
        closest = self.tree.query(self.features[img_index], k=k_neighbors, return_distance=False)

        return list(itertools.chain(*closest))

    def get_images(self, indexes):
        images = []

        for index in indexes:
            images.append(self.images[index])

        return images

    def save(self, filename):
        with h5py.File(filename,'w') as fw:
            fw['images'] = self.images
            fw['features'] = self.features
            fw['max_images'] = self.max_images
            fw['thumbnail_size'] = self.thumbnail_size

    def load(self, filename):
        with h5py.File(filename,'r') as fr:
            # load as list instead of numpy array
            self.images = []
            for img in fr['images']:
                self.images.append(img)

            self.features = []
            for feat in fr['features']:
                self.features.append(feat)

            # rebuild KDTree
            self.tree = None
            self.tree = KDTree(self.features, metric='euclidean')

            # load as list
            thumbnail_size = fr['thumbnail_size']
            self.thumbnail_size = (thumbnail_size[0], thumbnail_size[1], thumbnail_size[2])

            # reading scalar dataset
            data = fr['max_images']
            self.max_images = data[()]