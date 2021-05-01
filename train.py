import os
import time
import yaml
import numpy as np
import math
import argparse
from multiprocessing import Process

from typing import Tuple, Dict

EPOCHS = 40

def append_in_csv(filename: str, row: str):
    with open(filename, 'a') as f:
        f.write(row+'\n')

def load_data(check: bool) -> Tuple[np.ndarray]:
    from settings import BASE_DIR, BSD300_DIR

    from denoising.datasets import load_bsd300
    from denoising.datasets import load_dataset
    from denoising.datasets import extract_patches
    from denoising.datasets import add_noise

    imgs = load_bsd300(BSD300_DIR)
    patches = extract_patches(imgs, begin=(0,0), stride=10,
        dimension=(52,52), quantity_per_image=(5,5))
    
    y_train, y_test = load_dataset(patches, shuffle=False, split=(80,20))

    mean = 0.0
    variance = 0.01
    std = math.sqrt(variance) # 0.1

    x_train = add_noise(y_train, noise='gaussian', mean=mean, var=variance)
    x_test = add_noise(y_test, noise='gaussian', mean=mean, var=variance)

    x_train = x_train.astype('float32') / 255.0
    y_train = y_train.astype('float32') / 255.0

    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.astype('float32') / 255.0

    if check:
        return (x_train[:10], y_train[:10], x_test[:10], y_test[:10])

    return (x_train, y_train, x_test, y_test)

def load_config(file: str) -> Dict[str, str]:
    from settings import BASE_DIR

    config = {}

    with open(file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['output_path'] = os.path.join(BASE_DIR, config.get('output', 'results'), config.get('name', 'default'))
    os.makedirs(config['output_path'], exist_ok=True)

    config['metadata_path'] = os.path.join(config['output_path'], ".metadata")
    os.makedirs(config['metadata_path'], exist_ok=True)

    return config

def get_epochs(check):
    return EPOCHS if not check else 1

def train_mlp(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.mlp import MLP

    (x_train, y_train, _, _) = load_data(check)

    mlp = MLP(image_dimension=(52,52), hidden_layers=3, depth=32, multiply=True)
    mlp.compile(optimizer="adam", learning_rate=0.0001, loss="mse")

    ckpt = os.path.join(config['metadata_path'], f'MLP.hdf5')
    mlp.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    mlp.fit(epochs=get_epochs(check), batch_size=128, shuffle=True, 
        x_train=x_train, y_train=y_train,
        extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time

    mlp.save_loss_plot(os.path.join(config['output_path'], f'mlp_loss.png'))
    
    append_in_csv(config['time_file'], f'mlp, {time_in_seconds}')

def train_cnn(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.cnn import CNN

    (x_train, y_train, _, _) = load_data(check)

    cnn = CNN(image_dimension=(52,52), hidden_layers=10, depth=32, multiply=False, pooling=None)
    cnn.compile(optimizer="adam", learning_rate=0.0001, loss='mse')
    
    ckpt = os.path.join(config['metadata_path'], f'CNN.hdf5')
    cnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    cnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)
    time_in_seconds = time.time() - start_time

    cnn.save_loss_plot(os.path.join(config['output_path'], f'cnn_loss.png'))

    append_in_csv(config['time_file'], f'cnn, {time_in_seconds}')

def train_autoencoder(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.denoising_autoencoder import DenoisingAutoencoder

    (x_train, y_train, _, _) = load_data(check)

    ae = DenoisingAutoencoder(image_dimension=(52,52))
    ae.compile(optimizer='adam', learning_rate=1e-3, loss='mse')
    ckpt = os.path.join(config['metadata_path'], f'Autoencoder.hdf5')
    ae.set_checkpoint(ckpt, save_best_only=True, save_weights_only=False)

    ae.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    ae.save_loss_plot(os.path.join(config['output_path'], f'Autoencoder_loss.png'))

    append_in_csv(config['time_file'], f'autoencoder, {time_in_seconds}')


def train_cgan(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.cgan_denoiser.main import CGanDenoiser

    (x_train, y_train, _, _) = load_data(check)
    
    gan = CGanDenoiser(image_dimensions=(52,52))
    gan.compile(optimizer='adam', learning_rate=0.0001, loss='mse')
    gan.set_checkpoint(directory=os.path.join(config['metadata_path'], 'cgan-ckpt'))

    gan.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=256)
    time_in_seconds = time.time() - start_time

    append_in_csv(config['time_file'], f'gan, {time_in_seconds}')


def train_dncnn(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    (x_train, y_train, _, _) = load_data(check)

    dncnn = DnCNN(number_of_layers=19)
    dncnn.compile(optimizer="adam", learning_rate=0.0001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'DnCNN.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)
    dncnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time()
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN_loss.png'))

    append_in_csv(config['time_file'], f'dncnn, {time_in_seconds}')

def train(filename: str, check: bool):
    config = load_config(filename)

    config['time_file'] = os.path.join(config['metadata_path'], 'time.csv')
    append_in_csv(config['time_file'], 'name, seconds') 

    for func in [train_mlp, train_cnn, train_autoencoder, train_cgan, train_dncnn]:
        p = Process(target=func, args=(filename, check, config, ))
        p.start()
        p.join()
