import os, unittest, math
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage.util import random_noise

from denoising.methods.neural_network.deep_image_prior.main import DeepImagePrior
from denoising.methods.neural_network.deep_image_prior import deep_image_prior

from denoising.methods.neural_network.deep_image_prior.utils.denoising_utils import *

from settings import BASE_DIR, BSD300_DIR

from denoising.datasets import load_bsd300
from denoising.datasets import load_dataset
from denoising.datasets import extract_patches
from denoising.datasets import add_noise

from denoising.metrics import psnr, ssim

from denoising.utils import normalize


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
        
class TestDeepImagePriorWrapper(unittest.TestCase):
    def load_data(self):
        imgs = load_bsd300(BSD300_DIR)
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=(52,52), quantity_per_image=(5,5))
    
        _, y_test = load_dataset(patches, shuffle=False, split=(80,20))

        self.mean = 0.0
        self.variance = 0.01
        self.std = math.sqrt(self.variance) # 0.1

        x_test = add_noise(y_test, noise='gaussian', mean=self.mean, var=self.variance)

        x_test = x_test.astype('float32') / 255.0

        y_test = y_test.astype('float32') / 255.0

        return (x_test, y_test)

    def test_deep_image_prior_wrapper(self):
        (x_test, y_test) = self.load_data()

        pred = deep_image_prior(x_test[:2], iterations=10)

        self.assertTrue(psnr(normalize(y_test[:2], data_type='int'), normalize(pred, data_type='int')).mean() > 10)
        

if __name__ == '__main__':
    unittest.main()
