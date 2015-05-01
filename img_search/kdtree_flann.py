__author__ = 'martin.majer'

from pyflann import *
from scipy.misc import imsave
import cv2
import h5py
import itertools
import os.path
import numpy as np


class ImageSearchKDTreeFlann(object):
    def __init__(self, storage_dir, max_images=100000, thumbnail_size=(150, 150, 3)):
        self.filename = 'data.hdf5'
        self.tree = 'kdtree'
        self.thumbs = 'thumbs'
        self.storage_dir = storage_dir
        self.data_path = os.path.join(storage_dir, self.filename)
        self.tree_path = os.path.join(storage_dir, self.tree)
        self.max_images = max_images
        self.thumbnail_size = thumbnail_size
        self.features = []

        self.flann = FLANN()
        set_distance_type('euclidean', order = 0)

        if not os.path.exists(storage_dir):
            print 'Creating directory...'
            os.makedirs(storage_dir)
            os.makedirs(os.path.join(storage_dir, self.thumbs))
            print 'Directory created.'

        if os.path.isfile(self.data_path):
            print 'Loading data file...'
            self.load()
            print 'Data file loaded.'
        else:
            print 'Creating data file...'
            self.save(save_tree=False)
            print 'Data file created.'

    def add_images(self, images, features, build_tree=True):
        '''
        Add images, their features and calculate KDTree.
        :param images: list of images
        :param features: list of features
        :return: nothing
        '''
        if (len(self.features) + len(features)) > self.max_images:
            raise ValueError(
                'You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.features)))

        # add features
        self.features.extend(features)
        end = len(self.features)

        # save resized images
        dim = (self.thumbnail_size[0], self.thumbnail_size[1])

        for i, img in enumerate(images):
            img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
            img_resized = img_resized.astype(np.uint8)
            index = str(end - len(images) + i)
            print '\rAdding image #%s' % index,
            imsave(os.path.join(self.storage_dir, self.thumbs, index) + '.jpg', img_resized)

        if build_tree:
            self.build_tree()

    def build_tree(self):
        '''
        Build KDTree.
        :return: nothing
        '''
        print '\nCalculating KDTree...'
        self.flann.build_index(np.array(self.features), algorithm='kdtree')
        print 'Calculated.'

    def find_k_nearest_by_index(self, img_index, k_neighbors=3):
        '''
        Find K nearest neighbors by image index.
        :param img_index: index of searched image
        :param k: neighbors count
        :return: k nearest neighbors
        '''
        nearest, distances = self.flann.nn_index(self.features[img_index], k_neighbors)

        return list(itertools.chain.from_iterable(nearest)), list(itertools.chain.from_iterable(distances))

    def get_images(self, indexes):
        '''
        Get images on corresponding indexes.
        :param indexes: indexes of images
        :return: images
        '''
        images = []

        for index in indexes:
            img = cv2.imread(os.path.join(self.storage_dir, self.thumbs, str(index)) + '.jpg')
            images.append(img)

        return images

    def remove_last_image(self):
        '''
        Remove last image from features and database.
        :return: nothing
        '''
        last = len(self.features) - 1

        image = os.path.join(self.storage_dir, 'thumbs', str(last) + '.jpg')
        os.remove(image)

        print 'Removing image #%s' % last
        del self.features[-1]

    def save(self, save_tree=True):
        '''
        Save object variables to HDF5 and save KDTree.
        :return: nothing
        '''
        with h5py.File(self.data_path, 'w') as fw:
            fw['features'] = self.features
            fw['filename'] = self.filename
            fw['thumbs'] = self.thumbs
            fw['data_path'] = self.data_path
            fw['tree_path'] = self.tree_path
            fw['storage_dir'] = self.storage_dir
            fw['max_images'] = self.max_images
            fw['thumbnail_size'] = self.thumbnail_size
            fw['tree'] = self.tree

        if save_tree:
            self.flann.save_index(self.tree_path)

    def load(self):
        '''
        Load variables from HDF5 file and load KDTree.
        :return: nothing
        '''
        with h5py.File(self.data_path, 'r') as fr:
            # load features as list
            for feat in fr['features']:
                self.features.append(feat)

            # reading scalar dataset
            data = fr['filename']
            self.filename = data[()]
            data = fr['tree']
            self.tree = data[()]
            data = fr['thumbs']
            self.thumbs = data[()]
            data = fr['data_path']
            self.data_path = data[()]
            data = fr['tree_path']
            self.tree_path = data[()]
            data = fr['storage_dir']
            self.storage_dir = data[()]
            data = fr['max_images']
            self.max_images = data[()]

            # load as list and save as tuple
            thumbnail_size = fr['thumbnail_size']
            self.thumbnail_size = (thumbnail_size[0], thumbnail_size[1], thumbnail_size[2])

            # load KDTree
            self.flann.load_index(self.tree_path, np.array(self.features))