import os, unittest, math, time
import numpy as np
import pandas as pd
from skimage.measure import compare_psnr, compare_ssim
from skimage.util import random_noise

from denoising.methods.neural_network.cgan_denoiser.main import CGanDenoiser, train_cgan_denoiser

from denoising.methods.neural_network.deep_image_prior.utils.denoising_utils import *

from settings import BASE_DIR, BSD300_DIR

from denoising.datasets import load_bsd300
from denoising.datasets import load_dataset
from denoising.datasets import extract_patches
from denoising.datasets import add_noise

from denoising.metrics import psnr, ssim

from denoising.utils import normalize
        
class TestCGanDenoiser(unittest.TestCase):
    def setUp(self):
        self.data = self.load_data()

    def load_data(self):
        imgs = load_bsd300(BSD300_DIR)
        # patches = extract_patches(imgs, begin=(0,0), stride=10,
        #     dimension=(52,52), quantity_per_image=(5,5))
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=(28,28), quantity_per_image=(20,20))
        
        y_train, y_test = load_dataset(patches, shuffle=False, split=(80,20))

        self.mean = 0.0
        self.variance = 0.01
        self.std = math.sqrt(self.variance) # 0.1

        x_train = add_noise(y_train, noise='gaussian', mean=self.mean, var=self.variance)
        x_test = add_noise(y_test, noise='gaussian', mean=self.mean, var=self.variance)

        x_train = x_train.astype('float32') / 255.0
        y_train = y_train.astype('float32') / 255.0

        x_test = x_test.astype('float32') / 255.0
        y_test = y_test.astype('float32') / 255.0

        return (x_train[:60000], y_train[:60000], x_test[:10000], y_test[:10000])
    
    def test_load_cnn(self):
        # gan = CGanDenoiser(image_dimensions=(28,28))
        
        # gan.compile(optimizer='adam', learning_rate=0.0001, loss='mse')
        # gan.set_checkpoint()

        # gan.summary()

        (x_train, y_train, x_test, y_test) = self.data
        
        # gan.fit(epochs=20, x_train=x_train, y_train=y_train, batch_size=256)
        train_cgan_denoiser(epochs=20, x_train=x_train, y_train=y_train, batch_size=256)

    @unittest.skip
    def test_improvement(self):
        cnn = CNN(image_dimension=(52,52), 
            hidden_layers=10, 
            depth=32, 
            multiply=False,
            pooling=None)
        cnn.compile(optimizer='adam', learning_rate=1e-3, loss='mse')
        cnn.summary()

        (x_train, y_train, x_test, y_test) = self.data
        
        before = cnn.test(x_test)
        mean_before = psnr(normalize(y_test, data_type='int'), normalize(before, data_type='int')).mean()

        cnn.fit(epochs=10, x_train=x_train, y_train=y_train,  batch_size=32, shuffle=True, extract_validation_dataset=True)
        
        after = cnn.test(x_test)
        mean_after = psnr(normalize(y_test, data_type='int'), normalize(after, data_type='int')).mean()

        self.assertEquals(mean_after > mean_before)
        
if __name__ == '__main__':
    unittest.main()
