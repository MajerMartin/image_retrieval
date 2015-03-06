import unittest
import numpy as np
import h5py
import matplotlib.pyplot as plt

import img_search.distance_matrix


h5_imgs_fn = '/Users/martin.majer/PycharmProjects/PR4/data/sun_sample.hdf5'
h5_fts_fn = h5_imgs_fn + '.features.hdf5'

with h5py.File(h5_fts_fn,'r') as fr_features, h5py.File(h5_imgs_fn,'r') as fr_imgs:
    features = np.copy(fr_features['score'])
    imgs = np.array(fr_imgs['imgs'][:7][:,:,::-1])
    imgs = imgs.astype(np.float32) * 255

class DistanceMatrixTestClass(unittest.TestCase):

    def test_distance_matrix_init(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        self.assertEqual(dm.max_images, 100000)
        self.assertEqual(dm.thumbnail_size, (150,150,3))
        self.assertEqual(dm.distance_matrix, None)
        self.assertEqual(dm.images, [])
        self.assertEqual(dm.features, [])

    def test_distance_matrix_add_images(self):

        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        dm.add_images(imgs[:3,:,::-1], features[:3,:])
        self.assertEqual(dm.distance_matrix.shape, (3,3))
        self.assertEqual(len(dm.features), 3)
        self.assertEqual(len(dm.images), 3)

        dm.add_images(imgs[3,:,::-1], features[3,:])
        self.assertEqual(dm.distance_matrix.shape, (4,4))
        self.assertEqual(len(dm.features), 4)
        self.assertEqual(len(dm.images), 4)

        dm.add_images(imgs[4:6,:,::-1], features[4:6,:])
        self.assertEqual(dm.distance_matrix.shape, (6,6))
        self.assertEqual(len(dm.features), 6)
        self.assertEqual(len(dm.images), 6)

    def test_distance_matrix_get_images(self):

        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        dm.add_images(imgs[:6,:,::-1], features[:6,:])

        images = dm.get_images([1,3,5])
        self.assertEqual(images[0].shape, (150, 150, 3))

    def test_distance_matrix_find_k_nearest_by_index(self):

        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        dm.add_images(imgs[:6,:,::-1], features[:6,:])

        indexes = dm.find_k_nearest_by_index(1)
        self.assertEqual(len(indexes), 3)

if __name__ == '__main__':
    unittest.main()