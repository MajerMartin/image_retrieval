import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
import h5py
import shutil
from scipy.spatial.distance import squareform

import img_search.distance_matrix


class DistanceMatrixTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h5_imgs_fn = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sun_sample.hdf5')
        h5_fts_fn = h5_imgs_fn + '.features.hdf5'

        with h5py.File(h5_fts_fn, 'r') as fr_features, h5py.File(h5_imgs_fn, 'r') as fr_imgs:
            cls.features = np.copy(fr_features['score'])
            imgs = np.array(fr_imgs['imgs'][:7][:, :, ::-1])
            cls.imgs = imgs.astype(np.float32) * 255

        cls.storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm_test')

    @classmethod
    def tearDown(cls):
        shutil.rmtree(cls.storage_dir)

    def test_distance_matrix_init(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        self.assertEqual(dm.max_images, 100000)
        self.assertEqual(dm.thumbnail_size, (150, 150, 3))
        self.assertEqual(dm.distance_matrix, None)
        self.assertEqual(dm.features, [])
        self.assertEqual(dm.filename, 'data_dm.hdf5')
        self.assertEqual(dm.thumbs, 'thumbs')
        self.assertEqual(dm.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm_test', 'data_dm.hdf5'))
        self.assertEqual(dm.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm_test'))

    def test_distance_matrix_add_images(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        dm.add_images(self.imgs[:3, :, ::-1], self.features[:3, :])
        self.assertEqual(dm.distance_matrix.shape, (3, 3))
        self.assertEqual(len(dm.features), 3)

        dm.add_images([self.imgs[3, :, ::-1]], [self.features[3, :]])
        self.assertEqual(dm.distance_matrix.shape, (4, 4))
        self.assertEqual(len(dm.features), 4)

        dm.add_images(self.imgs[4:6, :, ::-1], self.features[4:6, :])
        self.assertEqual(dm.distance_matrix.shape, (6, 6))
        self.assertEqual(len(dm.features), 6)

    def test_distance_matrix_get_images(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        dm.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        images = dm.get_images([1, 3, 5])
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].shape, (150, 150, 3))

        # for img in images:
        # plt.figure()
        # plt.imshow(img[:,:,::-1])
        #     plt.show()

    def test_distance_matrix_find_k_nearest_by_index(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        dm.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        indexes = dm.find_k_nearest_by_index(1)
        self.assertEqual(len(indexes), 3)

    def test_distance_matrix_load(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        dm.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])
        dm.save()

        dm_copy = img_search.distance_matrix.ImageSearchDistanceMatrix(self.storage_dir)

        self.assertEqual(len(dm_copy.features), 6)
        self.assertEqual(dm_copy.features[0].shape, (1000,))
        self.assertEqual(dm_copy.distance_matrix.shape, (6, 6))
        self.assertEqual(dm_copy.thumbnail_size, (150, 150, 3))
        self.assertEqual(dm_copy.max_images, 100000)
        self.assertEqual(dm_copy.filename, 'data_dm.hdf5')
        self.assertEqual(dm_copy.thumbs, 'thumbs')
        self.assertEqual(dm_copy.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm_test', 'data_dm.hdf5'))
        self.assertEqual(dm_copy.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm_test'))
        self.assertEqual(squareform(dm.distance_matrix).all(), squareform(dm_copy.distance_matrix).all())


if __name__ == '__main__':
    unittest.main()