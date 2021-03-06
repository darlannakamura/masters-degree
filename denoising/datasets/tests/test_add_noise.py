import unittest
import numpy as np
from settings import BSD300_DIR
from denoising.datasets import load_bsd300, add_noise
from denoising.metrics import psnr, ssim

class TestAddNoise(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        nparr = load_bsd300(BSD300_DIR)
        self.arr = nparr[:10]

    def test_invalid_np_array_type(self):
        with self.assertRaises(AssertionError):
            add_noise([], noise='poisson')

    def test_invalid_np_array_dimensions(self):
        with self.assertRaises(AssertionError):
            add_noise(np.random.rand(100,50,50), noise='poisson')

    def test_invalid_noise(self):
        with self.assertRaises(AssertionError):
            add_noise(self.arr, noise='speckle')

    def test_add_gaussian_without_mean(self):
        with self.assertRaises(AssertionError):
            add_noise(self.arr, noise='gaussian', var=0.01)

    def test_add_gaussian_without_var(self):
        with self.assertRaises(AssertionError):
            add_noise(self.arr, noise='gaussian', mean=0.0)

    def test_add_gaussian_with_str_mean(self):
        with self.assertRaises(AssertionError):
            add_noise(self.arr, noise='gaussian', mean='0.0', var=0.01)

    def test_if_gaussian_noisy_psnr_is_lower_than_100dB(self):
        noisy = add_noise(self.arr, noise='gaussian', mean=0.0, var=1.0)
        psnr_arr = psnr(self.arr, noisy)
        self.assertTrue(psnr_arr.mean() < 100.0)

    def test_if_gaussian_noisy_is_lower_than_dot_fifty_ssim(self):
        noisy = add_noise(self.arr, noise='gaussian', mean=0.0, var=1.0)
        ssim_arr = ssim(self.arr, noisy)
        self.assertTrue(np.array(ssim_arr).mean() < .5)

    def test_if_poisson_noisy_psnr_is_lower_than_100dB(self):
        noisy = add_noise(self.arr, noise='poisson')
        psnr_arr = psnr(self.arr, noisy)
        self.assertTrue(np.array(psnr_arr).mean() < 100.0)

    def test_if_poisson_noisy_is_lower_than_dot_eighty_ssim(self):
        noisy = add_noise(self.arr, noise='poisson')
        ssim_arr = ssim(self.arr, noisy)
        self.assertTrue(np.array(ssim_arr).mean() < .8)

if __name__ == '__main__':
    unittest.main()
