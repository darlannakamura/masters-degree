import os
import time
import yaml
import numpy as np
from typing import Tuple, Dict
from multiprocessing import Process
from denoising.metrics import psnr, ssim
from denoising.utils import normalize

from train import *

def train_dncnn(layers: int, samples: int, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    (x_train, y_train, x_test, y_test) = load_data(check=False, config=config)
    
    x_train = x_train[:samples]
    y_train = y_train[:samples]

    dncnn = DnCNN(number_of_layers=layers)
    dncnn.compile(optimizer="adam", learning_rate=0.0001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'DnCNN{layers}_{samples}.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    dncnn.fit(epochs=get_epochs(False), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = round(time.time() - start_time,2)
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN{layers}_{samples}_loss.png'))

    dncnn.load(ckpt)
    predicted = dncnn.test(x_test)

    psnr_data = round(psnr(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)
    
    ssim_data = round(ssim(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)

    append_in_csv(config['comparison_file'], f'{layers}, {samples}, {psnr_data}, {ssim_data}, {time_in_seconds}')


def train_mlp(layers: int, samples: int, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.mlp import MLP

    (x_train, y_train, x_test, y_test) = load_data(check=False, config=config)
    
    x_train = x_train[:samples]
    y_train = y_train[:samples]

    mlp = MLP(image_dimension=(50,50), hidden_layers=layers, depth=32, multiply=False)
    mlp.compile(optimizer="adam", learning_rate=0.0001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'MLP{layers}_{samples}.hdf5')
    mlp.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    mlp.fit(epochs=get_epochs(False), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = round(time.time() - start_time, 2)
    mlp.save_loss_plot(os.path.join(config['output_path'], f'MLP{layers}_{samples}_loss.png'))

    mlp.load(ckpt)
    predicted = mlp.test(x_test)

    psnr_data = round(psnr(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)
    
    ssim_data = round(ssim(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)

    append_in_csv(config['comparison_file'], f'{layers}, {samples}, {psnr_data}, {ssim_data}, {time_in_seconds}')


def train_cnn(layers: int, samples: int, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.cnn import CNN

    (x_train, y_train, x_test, y_test) = load_data(check=False, config=config)
    
    x_train = x_train[:samples]
    y_train = y_train[:samples]

    cnn = CNN(image_dimension=(50,50), hidden_layers=layers, depth=32, multiply=False, pooling=None)
    cnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'CNN{layers}_{samples}.hdf5')
    cnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    cnn.fit(epochs=get_epochs(False), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = round(time.time() - start_time, 2)
    cnn.save_loss_plot(os.path.join(config['output_path'], f'CNN{layers}_{samples}_loss.png'))

    cnn.load(ckpt)
    predicted = cnn.test(x_test)

    psnr_data = round(psnr(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)
    
    ssim_data = round(ssim(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)

    append_in_csv(config['comparison_file'], f'{layers}, {samples}, {psnr_data}, {ssim_data}, {time_in_seconds}')


def train_autoencoder(samples: int, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.denoising_autoencoder import DenoisingAutoencoder

    (x_train, y_train, x_test, y_test) = load_data(check=False, config=config)
    
    x_train = x_train[:samples]
    y_train = y_train[:samples]

    cnn = DenoisingAutoencoder(image_dimension=(52,52))
    cnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'Autoencoder_{samples}.hdf5')
    cnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    cnn.fit(epochs=get_epochs(False), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = round(time.time() - start_time, 2)
    cnn.save_loss_plot(os.path.join(config['output_path'], f'Autoencoder_{samples}_loss.png'))

    cnn.load(ckpt)
    predicted = cnn.test(x_test)

    psnr_data = round(psnr(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)
    
    ssim_data = round(ssim(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)

    append_in_csv(config['comparison_file'], f'{samples}, {psnr_data}, {ssim_data}, {time_in_seconds}')

def train_cgan(samples: int, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.cgan_denoiser.main import CGanDenoiser

    (x_train, y_train, x_test, y_test) = load_data(check=False, config=config)
    
    x_train = x_train[:samples]
    y_train = y_train[:samples]

    cnn = CGanDenoiser(image_dimensions=(50,50))
    cnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'CGAN_{samples}.hdf5')
    cnn.set_checkpoint(directory=os.path.join(config['metadata_path'], 'cgan-ckpt'))

    cnn.fit(epochs=get_epochs(False), x_train=x_train, y_train=y_train, batch_size=256)

    time_in_seconds = round(time.time() - start_time, 2)
    # cnn.save_loss_plot(os.path.join(config['output_path'], f'CGAN_{samples}_loss.png'))

    # cnn.load(ckpt)
    predicted = cnn.test(x_test)

    psnr_data = round(psnr(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)
    
    ssim_data = round(ssim(
        normalize(y_test, data_type='int'), 
        normalize(predicted, data_type='int')
    ).mean(), 2)

    append_in_csv(config['comparison_file'], f'{samples}, {psnr_data}, {ssim_data}, {time_in_seconds}')


def compare(filename: str, method: str, check: bool):
    config = load_config(filename)

    config['comparison_file'] = os.path.join(config['metadata_path'], f'{method}_comparison.csv')
    append_in_csv(config['comparison_file'], 'layers, samples, psnr, ssim, seconds')

    if method.lower() == 'dncnn':
        method_func = train_dncnn

        for layer in [10, 15, 19]:
            for samples in [5,10, 15]:
                samples *= 1000

                p = Process(target=method_func, args=(layer, samples, config, ))
                p.start()
                p.join()

    elif method.lower() == 'mlp':
        method_func = train_mlp

        for layer in [3, 5,10,15,20, 25]:
            for samples in [5, 10, 15]:
                samples *= 1000

                p = Process(target=method_func, args=(layer, samples, config, ))
                p.start()
                p.join()

    elif method.lower() == 'cnn':
        method_func = train_cnn

        for layer in [3, 5, 10, 15, 20]:
            for samples in [5,10,15]:
                samples *= 1000

                p = Process(target=method_func, args=(layer, samples, config, ))
                p.start()
                p.join()

    elif method.lower() == 'autoencoder':
        method_func = train_autoencoder

        for samples in [5,10,15]:
            samples *= 1000

            p = Process(target=method_func, args=(samples, config, ))
            p.start()
            p.join()

    elif method.lower() == 'cgan':
        method_func = train_cgan

        for samples in [5,10,15]:
            samples *= 1000

            p = Process(target=method_func, args=(samples, config, ))
            p.start()
            p.join()