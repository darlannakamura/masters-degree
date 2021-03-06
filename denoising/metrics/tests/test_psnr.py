import os
import cv2
import unittest
import numpy as np
from copy import deepcopy
from denoising.metrics import psnr

class TestPSNR(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.arr = np.random.randint(low=0, high=255, size=(10, 500, 500, 1))

    def test_psnr_with_two_equal_sets(self):
        avg_psnr = np.array(psnr(self.arr, deepcopy(self.arr))).mean()
        self.assertEqual(avg_psnr, float('inf'))

    def test_psnr_with_two_completely_different_sets(self):
        """Test PSNR of 10 full black images comparing with 10 full white images.
        """
        low = np.zeros((10, 500, 500, 1), dtype=np.uint8)
        high = np.ones((10, 500, 500, 1), dtype=np.uint8) * 255

        avg_psnr = np.array(psnr(high, low)).mean()
        self.assertEqual(avg_psnr, 0.0)

if __name__ == '__main__':
    unittest.main()
