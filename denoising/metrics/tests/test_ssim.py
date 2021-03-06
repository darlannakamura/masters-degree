import os
import cv2
import unittest
import numpy as np
from copy import deepcopy
from denoising.metrics import psnr

class TestSSIM(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.arr = np.random.randint(low=0, high=255, size=(10, 500, 500, 1))

    def test_ssim_with_two_equal_sets(self):
        avg_ssim = np.array(ssim(self.arr, deepcopy(self.arr))).mean()
        self.assertEqual(avg_ssim, 1.0)

    def test_ssim_with_two_completely_different_sets(self):
        """Test SSIM of 10 full black images comparing with 10 full white images.
        """
        low = np.zeros((10, 500, 500, 1), dtype=np.uint8)
        high = np.ones((10, 500, 500, 1), dtype=np.uint8) * 255

        avg_ssim = np.array(psnr(high, low)).mean()
        self.assertEqual(avg_ssim, 0.0)

if __name__ == '__main__':
    unittest.main()
