import unittest

import img_search.distance_matrix

class DistanceMatrixTestClass(unittest.TestCase):
    def test_distance_matrix_init(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

    def test_distance_matrix_get_images(self):
        dm = img_search.distance_matrix.ImageSearchDistanceMatrix()

        images = dm.get_images([])

        self.assertEqual(images, [])

if __name__ == '__main__':
    unittest.main()