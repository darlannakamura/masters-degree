import os, unittest, math, time
import numpy as np
import pandas as pd
from skimage.measure import compare_psnr, compare_ssim
from skimage.util import random_noise

from denoising.methods.neural_network.mlp import MLP

from denoising.methods.neural_network.deep_image_prior.utils.denoising_utils import *

from settings import BASE_DIR, BSD300_DIR

from denoising.datasets import load_bsd300
from denoising.datasets import load_dataset
from denoising.datasets import extract_patches
from denoising.datasets import add_noise

from denoising.metrics import psnr, ssim

from denoising.utils import normalize


# class TestMLP(unittest.TestCase):        
    # def test_improvement_with_deep_image_prior(self):
    #     filename = os.path.join(BASE_DIR, 'denoising', 'methods', 'neural_network', 'deep_image_prior', 'data', 'cameraman.png')
        
    #     # Add synthetic noise
    #     img_pil = crop_image(get_image(filename, -1)[0], d=32)
    #     img_np = pil_to_np(img_pil)

    #     img_noisy_np = random_noise(img_np, mode='poisson').astype(np.float32)

    #     dip = DeepImagePrior()

    #     out, out_avg = dip.run(iterations=10, image_noisy=img_noisy_np, noise_std_dev=1.0)

    #     self.assertTrue(compare_psnr(img_np, out) > 10)
    #     self.assertTrue(compare_psnr(img_np, out_avg) > 10)

    #     img_np = np.squeeze(np.moveaxis(img_np[...,0],0,-1))
    #     out_np = np.squeeze(np.moveaxis(out[...,0],0,-1))
    #     out_np_avg = np.squeeze(np.moveaxis(out_avg[...,0],0,-1))

    #     self.assertTrue(compare_ssim(img_np, out_np) > 0.50)
    #     self.assertTrue(compare_ssim(img_np, out_np_avg) > 0.50)
        
class TestMLP(unittest.TestCase):
    def setUp(self):
        self.data = self.load_data()

    def load_data(self):
        imgs = load_bsd300(BSD300_DIR)
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=(52,52), quantity_per_image=(5,5))
    
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

        return (x_train[:100], y_train[:100], x_test[:100], y_test[:100])

    @unittest.skip
    def test_mlp(self):
        mlp = MLP(image_dimension=(52,52), hidden_layers=3, depth=32, multiply=True)
        mlp.compile(optimizer='adam', learning_rate=0.0001, loss='mse')
        mlp.summary()

        (x_train, y_train, x_test, y_test) = self.data
        
        mlp.fit(epochs=1, x_train=x_train, y_train=y_train,  batch_size=2, shuffle=True, extract_validation_dataset=True)
    
    def test_improvement(self):
        mlp = MLP(image_dimension=(52,52), hidden_layers=3, depth=32, multiply=True)
        mlp.compile(optimizer='adam', learning_rate=1e-3, loss='mse')
        mlp.summary()

        (x_train, y_train, x_test, y_test) = self.data
        
        pred = mlp.test(x_test)
        print('PSNR before training: ', psnr(normalize(y_test, data_type='int'), normalize(pred, data_type='int')).mean())

        mlp.fit(epochs=10, x_train=x_train, y_train=y_train,  batch_size=32, shuffle=True, extract_validation_dataset=True)
        pred = mlp.test(x_test)
        print('PSNR after 10 epochs: ', psnr(normalize(y_test, data_type='int'), normalize(pred, data_type='int')).mean())

        mlp.fit(epochs=30, x_train=x_train, y_train=y_train,  batch_size=32, shuffle=True, extract_validation_dataset=True)
        pred = mlp.test(x_test)
        print('PSNR after 40 epochs: ', psnr(normalize(y_test, data_type='int'), normalize(pred, data_type='int')).mean())

        # self.assertTrue(psnr(normalize(y_test[:2], data_type='int'), normalize(pred, data_type='int')).mean() > 10)


#     def test_deep_image_prior_wrapper(self):
#         (x_test, y_test) = self.data

#         pred = deep_image_prior(x_test[:2], iterations=10, noise_std_dev=1.0)

#         self.assertTrue(psnr(normalize(y_test[:2], data_type='int'), normalize(pred, data_type='int')).mean() > 10)
    
#     @unittest.skip("Slow test. Usually takes more than 2 minutes to run.")
#     def test_improvement_on_deep_image_prior_wrapper(self):
#         (x_test, y_test) = self.data

#         pred = deep_image_prior(x_test[:2], iterations=10, noise_std_dev=1.0)
#         self.assertTrue(psnr(normalize(y_test[:2], data_type='int'), normalize(pred, data_type='int')).mean() > 10)

#         pred = deep_image_prior(x_test[:2], iterations=1000, noise_std_dev=1.0)
#         self.assertTrue(psnr(normalize(y_test[:2], data_type='int'), normalize(pred, data_type='int')).mean() > 20)
    
#     @unittest.skip("Too much slow test. Used to find the best balance between iterations and PSNR/SSIM.")
#     def test_to_find_the_best_iterations(self):
#         (x_test, y_test) = self.data

#         data = []

#         for it in [50, 100, 200, 500]:
#             start_time = time.time()
#             pred = deep_image_prior(x_test[:100], iterations=it, noise_std_dev=1.0)

#             psnr_avg = psnr(normalize(y_test[:100], data_type='int'), normalize(pred, data_type='int')).mean()
#             ssim_avg = ssim(normalize(y_test[:100], data_type='int'), normalize(pred, data_type='int')).mean()

#             data.append({
#                 'iterations': it,
#                 'time': time.time() - start_time,
#                 'psnr': psnr_avg,
#                 'ssim': ssim_avg
#             })

#             print(f'Iterations {it}: time (seconds): {time.time() - start_time }, psnr: {psnr_avg}, ssim: {ssim_avg}')

#         df = pd.DataFrame(data)
#         df.to_csv('deep_image_prior2.csv')


if __name__ == '__main__':
    unittest.main()
