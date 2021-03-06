import unittest
import numpy as np
from copy import deepcopy
from denoising.datasets import load_dataset

class TestLoadDataset(unittest.TestCase):
    def setUp(self):
        self. arr = self.load_random_np_array()

    def load_random_np_array(self):
        return np.random.rand(100, 50, 50, 1)

    def test_invalid_np_array_type(self):
        with self.assertRaises(AssertionError):
            load_dataset([])
    
    def test_invalid_np_array_dimension(self):
        with self.assertRaises(AssertionError):
            load_dataset(np.random.rand(100, 50, 50))

    def test_invalid_split_type(self):
        with self.assertRaises(AssertionError):
            load_dataset(self.arr, split=.8)

    def test_invalid_split_dimension(self):
        with self.assertRaises(AssertionError):
            load_dataset(self.arr, split=(70,10,20))

    def test_invalid_split_sum(self):
        with self.assertRaises(AssertionError):
            load_dataset(self.arr, split=(80,21))

    def test_split(self):
        train, test = load_dataset(self.arr, split=(66,34))
        self.assertEqual(train.shape[0] + test.shape[0], self.arr.shape[0])

    def test_shuffling(self):
        train, test = load_dataset(self.arr, shuffle=True, split=(80,20))
        self.assertEqual((train==self.arr[:80]).all(), False)

    def test_random_shuffling(self):
        train1, test1 = load_dataset(self.arr, shuffle=True, split=(80,20))
        train2, test2 = load_dataset(self.arr, shuffle=True, split=(80,20))
        
        self.assertEqual((train1==train2).all(), False)
        self.assertEqual((test1==test2).all(), False)
        
if __name__ == '__main__':
    unittest.main()
