__author__ = 'martin.majer'

import numpy as np
import cv2
import h5py
import os.path
from scipy.misc import imsave
from scipy.spatial.distance import squareform

filename = 'data_dm.hdf5'
thumbs = 'thumbs/'

class ImageSearchDistanceMatrix(object):
    def __init__(self, storage_dir, max_images=100000, thumbnail_size=(150,150,3)):
        self.storage_dir = storage_dir
        self.data_path = storage_dir + filename
        self.max_images = max_images
        self.thumbnail_size = thumbnail_size
        self.distance_matrix = None
        self.features = []

        if not os.path.exists(storage_dir):
            print 'Creating directory...'
            os.makedirs(storage_dir)
            os.makedirs(storage_dir + thumbs)
            print 'Directory created.'

        if os.path.isfile(self.data_path):
            print 'Loading data file...'
            self.load()
            print 'Data file loaded.'
        else:
            print 'Creating data file...'
            self.save()
            print 'Data file created.'

    def add_images(self, images, features):
        '''
        Save images, add their features and calculate distance matrix.
        :param images: list of images
        :param features: list of features
        :return: nothing
        '''

        if (len(self.features) + len(features)) > self.max_images:
            raise ValueError('You can add only %d more image(s). Maximum limit achieved.' % (self.max_images - len(self.features)))

        # add features
        self.features.extend(features)
        start = len(self.features) - len(features)
        end = len(self.features)

        # save resized images
        dim = (self.thumbnail_size[0], self.thumbnail_size[1])

        for i, img in enumerate(images):
            img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
            img_resized = img_resized.astype(np.uint8)
            index = str(end - len(images) + i)
            print '\rAdding image #%s' % index,
            imsave(self.storage_dir + thumbs + index + '.jpg', img_resized)

        # initialize new distance matrix
        if self.distance_matrix is None:
            self.distance_matrix = np.zeros((end, end), dtype=np.float16)

            # fill new distance matrix
            for i in range(0, end):
                for j in range(i, end):
                    self.distance_matrix[i,j] = np.linalg.norm(self.features[i] - self.features[j])

        # add rows and columns
        else:
            for i in range(start, end):
                new_row = np.zeros((1, i))
                self.distance_matrix = np.concatenate((self.distance_matrix, new_row), axis=0)

                new_col = np.zeros((i + 1, 1))

                for j in range(0, i):
                    new_col[j,0] = np.linalg.norm(self.features[i] - self.features[j])

                self.distance_matrix = np.concatenate((self.distance_matrix, new_col), axis=1)

        print '\nTransposing matrix...'
        self.distance_matrix += self.distance_matrix.T
        print 'Transposed.'

    def find_k_nearest_by_index(self, img_index, k=3):
        '''
        Find K nearest neighbors by image index.
        :param img_index: index of searched image
        :param k: neighbors count
        :return: k nearest neighbors
        '''
        row = self.distance_matrix[img_index,:]
        nearest = np.argsort(row)

        return nearest[:k]

    def get_images(self, indexes):
        '''
        Get images on corresponding indexes.
        :param indexes: indexes of images
        :return: images
        '''
        images = []

        for index in indexes:
            img = cv2.imread(self.storage_dir + thumbs + str(index) + '.jpg')
            images.append(img)

        return images

    def save(self):
        '''
        Save object variables to HDF5.
        :return: nothing
        '''
        with h5py.File(self.data_path,'w') as fw:
            fw['features'] = self.features
            fw['data_path'] = self.data_path
            fw['storage_dir'] = self.storage_dir
            fw['max_images'] = self.max_images
            fw['thumbnail_size'] = self.thumbnail_size

            # create data set for images and don't save distance matrix when creating hdf5 file
            if self.distance_matrix is not None:
                fw['distance_matrix'] = squareform(self.distance_matrix)

    def load(self):
        '''
        Load variables from HDF5 file.
        :return: nothing
        '''
        with h5py.File(self.data_path,'r') as fr:
            # load features as list
            for feat in fr['features']:
                self.features.append(feat)

            # reading scalar dataset
            data = fr['data_path']
            self.data_path = data[()]
            data = fr['storage_dir']
            self.storage_dir = data[()]
            data = fr['max_images']
            self.max_images = data[()]

            # load whole matrix if exists
            try:
                self.distance_matrix = squareform(fr['distance_matrix'])
            except:
                pass

            # load as list and save as tuple
            thumbnail_size = fr['thumbnail_size']
            self.thumbnail_size = (thumbnail_size[0], thumbnail_size[1], thumbnail_size[2])