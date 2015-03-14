__author__ = 'martin.majer'

import os
import unittest
import numpy as np
import matplotlib.pyplot as plt
import h5py
import shutil

import img_search.kdtree


class KDTreeTestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        h5_imgs_fn = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'sun_sample.hdf5')
        h5_fts_fn = h5_imgs_fn + '.features.hdf5'

        with h5py.File(h5_fts_fn, 'r') as fr_features, h5py.File(h5_imgs_fn, 'r') as fr_imgs:
            cls.features = np.copy(fr_features['score'])
            imgs = np.array(fr_imgs['imgs'][:7][:, :, ::-1])
            cls.imgs = imgs.astype(np.float32) * 255

        cls.storage_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_test')

    @classmethod
    def tearDown(cls):
        shutil.rmtree(cls.storage_dir)

    def test_kdtree_init(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        self.assertEqual(kdt.max_images, 100000)
        self.assertEqual(kdt.thumbnail_size, (150, 150, 3))
        self.assertEqual(kdt.tree, None)
        self.assertEqual(kdt.features, [])
        self.assertEqual(kdt.filename, 'data_kdt.hdf5')
        self.assertEqual(kdt.thumbs, 'thumbs')
        self.assertEqual(kdt.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_test', 'data_kdt.hdf5'))
        self.assertEqual(kdt.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_test'))

    def test_kdtree_add_images(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        kdt.add_images(self.imgs[:3, :, ::-1], self.features[:3, :])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 3)

        kdt.add_images([self.imgs[3, :, ::-1]], [self.features[3, :]])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 4)

        kdt.add_images(self.imgs[4:6, :, ::-1], self.features[4:6, :])
        self.assertNotEqual(kdt.tree, None)
        self.assertEqual(len(kdt.features), 6)

    def test_kdtree_get_images(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        images = kdt.get_images([1, 3, 5])
        self.assertEqual(len(images), 3)
        self.assertEqual(images[0].shape, (150, 150, 3))

        #for img in images:
        #    plt.figure()
        #    plt.imshow(img[:,:,::-1])
        #    plt.show()

        images = kdt.get_images([1])
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].shape, (150, 150, 3))

    def test_kdtree_find_k_nearest_by_index(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])

        indexes = kdt.find_k_nearest_by_index(1)
        self.assertEqual(len(indexes), 3)

    def test_kdtree_load(self):
        kdt = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        kdt.add_images(self.imgs[:6, :, ::-1], self.features[:6, :])
        kdt.save()

        kdt_copy = img_search.kdtree.ImageSearchKDTree(self.storage_dir)

        self.assertEqual(len(kdt_copy.features), 6)
        self.assertEqual(kdt_copy.features[0].shape, (1000,))
        self.assertNotEqual(kdt_copy.tree, None)
        self.assertEqual(kdt_copy.thumbnail_size, (150, 150, 3))
        self.assertEqual(kdt_copy.max_images, 100000)
        self.assertEqual(kdt_copy.filename, 'data_kdt.hdf5')
        self.assertEqual(kdt_copy.thumbs, 'thumbs')
        self.assertEqual(kdt_copy.data_path,
                         os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_test', 'data_kdt.hdf5'))
        self.assertEqual(kdt_copy.storage_dir, os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kdt_test'))


if __name__ == '__main__':
    unittest.main()