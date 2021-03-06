import os
import unittest
import numpy as np
from settings import BASE_DIR, BSD300_DIR
from denoising.datasets import load_bsd300
from denoising.datasets import DirectoryNotFoundError, LoadDatasetError

class TestLoadBSD300(unittest.TestCase):
    def load(self):
        self.arr = load_bsd300(BSD300_DIR)

    def test_load_bsd_from_unknown_directory_path(self):
        with self.assertRaises(DirectoryNotFoundError):
            load_bsd300("/tmp/UNKNOWN_RANDOM_DIR_DATASET")

    def test_load_bsd_from_a_directory_without_jpg_files(self):
        with self.assertRaises(LoadDatasetError):
            load_bsd300(os.path.join(BASE_DIR, "denoising", "datasets", "tests"))

    def test_load_bsd_as_numpy_array(self):
        self.load()
        self.assertEqual(type(self.arr), np.ndarray)
    
    def test_load_bsd_has_300_images(self):
        self.load()
        self.assertEqual(self.arr.shape[0], 300)        

    def test_load_bsd_has_four_dimensions(self):
        self.load()
        self.assertEqual(len(self.arr.shape), 4)
    
    def test_load_bsd_has_images_of_321_per_481_dimensions(self):
        self.load()

        self.assertEqual(self.arr.shape[1], 321)
        self.assertEqual(self.arr.shape[2], 481)
        
if __name__ == '__main__':
    unittest.main()
