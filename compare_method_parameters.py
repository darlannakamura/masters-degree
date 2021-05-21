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

    dncnn.fit(epochs=get_epochs(True), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
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
