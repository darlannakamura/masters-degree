import os, unittest
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.util import random_noise

from denoising.methods.neural_network.deep_image_prior import DeepImagePrior
from denoising.methods.neural_network.deep_image_prior.utils.denoising_utils import *

from settings import BASE_DIR

class TestDeepImagePrior(unittest.TestCase):
    def test_improvement_with_deep_image_prior(self):
        filename = os.path.join(BASE_DIR, 'denoising', 'methods', 'neural_network', 'deep_image_prior', 'data', 'cameraman.png')
        
        # Add synthetic noise
        img_pil = crop_image(get_image(filename, -1)[0], d=32)
        img_np = pil_to_np(img_pil)

        img_noisy_np = random_noise(img_np, mode='poisson').astype(np.float32)

        dip = DeepImagePrior()

        out, out_avg = dip.run(iterations=10, image_noisy=img_noisy_np)

        self.assertTrue(compare_psnr(img_np, out) > 10)
        self.assertTrue(compare_psnr(img_np, out_avg) > 10)

        img_np = np.squeeze(np.moveaxis(img_np[...,0],0,-1))
        out_np = np.squeeze(np.moveaxis(out[...,0],0,-1))
        out_np_avg = np.squeeze(np.moveaxis(out_avg[...,0],0,-1))

        self.assertTrue(compare_ssim(img_np, out_np) > 0.50)
        self.assertTrue(compare_ssim(img_np, out_np_avg) > 0.50)
        
if __name__ == '__main__':
    unittest.main()
