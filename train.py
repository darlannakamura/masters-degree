import os
import time
import yaml
import numpy as np
import math
import argparse
from multiprocessing import Process

from typing import Tuple, Dict

EPOCHS = 40
IMAGE_DIMENSION = (50,50)

def append_in_csv(filename: str, row: str):
    with open(filename, 'a') as f:
        f.write(row+'\n')

def load_data(check: bool, config: Dict[str, str])  -> Tuple[np.ndarray]:
    from copy import deepcopy
    import cv2
    from settings import BASE_DIR, BSD300_DIR, PROJECTIONS_DIR

    from denoising.datasets import load_bsd300
    from denoising.datasets import load_dataset
    from denoising.datasets import extract_patches
    from denoising.datasets import add_noise

    from denoising.utils import normalize


    dataset = config['dataset']

    if dataset.lower() == 'bsd300':
        imgs = load_bsd300(BSD300_DIR)
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=(50,50), quantity_per_image=(10,10))
        
        if 'shuffle' in config and config['shuffle']:
            np.random.seed(10)
            np.random.shuffle(patches)

        y_train, y_test = load_dataset(patches, shuffle=False, split=(80,20))

        noise = config['noise']

        if isinstance(noise, str):
            if noise == 'poisson':
                x_train = add_noise(y_train, noise='poisson')
                x_test = add_noise(y_test, noise='poisson')
        if isinstance(noise, dict):
            assert 'type' in noise, "noise should have 'type' attribute. Options are: gaussian and poisson-gaussian."
            noise_type = noise['type']
            assert 'mean' in noise, "noise should have 'mean' attribute."
            assert 'variance' in noise, "noise should have 'variance' attribute."

            mean = float(noise['mean'])
            variance = float(noise['variance'])

            if noise_type == 'gaussian':
                x_train = add_noise(y_train, noise='gaussian', mean=mean, var=variance)
                x_test = add_noise(y_test, noise='gaussian', mean=mean, var=variance)
            elif noise_type == 'poisson-gaussian':
                x_train = add_noise(y_train, noise='poisson')
                x_test = add_noise(y_test, noise='poisson')

                x_train = add_noise(y_train, noise='gaussian', mean=mean, var=variance)
                x_test = add_noise(y_test, noise='gaussian', mean=mean, var=variance)
    elif dataset.lower() == 'dbt':
        noisy_projections = np.load(os.path.join(PROJECTIONS_DIR, 'noisy_10.npy'))
        noisy_projections = noisy_projections.reshape((-1, 1792, 2048, 1))
        
        noisy_patches = extract_patches(noisy_projections, begin=(0,500), stride=10,
            dimension=(52,52), quantity_per_image=(10,10))

        x_train, x_test = load_dataset(noisy_patches, shuffle=False, split=(80,20))

        original_projections = np.load(os.path.join(PROJECTIONS_DIR, 'original_10.npy'))
        original_projections = original_projections.reshape((-1, 1792, 2048, 1))
        
        original_patches = extract_patches(original_projections, begin=(0,500), stride=10,
            dimension=(52,52), quantity_per_image=(10,10))

        y_train, y_test = load_dataset(original_patches, shuffle=False, split=(80,20))
    elif dataset.lower() == 'spie_2021':
        from denoising.datasets.spie_2021 import carrega_dataset, adiciona_a_dimensao_das_cores

        full_x_train, full_y_train, full_x_test, full_y_test = carrega_dataset(
          '/content/gdrive/My Drive/Colab Notebooks/dataset/patch-50x50-cada-projecao-200', 
          divisao=(80,20), embaralhar=True)

        x_train =  np.reshape(full_x_train, (-1, 50, 50))
        x_test = np.reshape(full_x_test, (-1, 50, 50))
        y_train = np.reshape(full_y_train, (-1, 50, 50))
        y_test = np.reshape(full_y_test, (-1, 50, 50)) 

        del full_x_train
        del full_y_train
        del full_x_test
        del full_y_test


        x_train = x_train[:15000]
        y_train = y_train[:15000]

        x_test = x_test[:3750]
        y_test = y_test[:3750]


        np.random.seed(13)
        np.random.shuffle(x_train)

        np.random.seed(13)
        np.random.shuffle(y_train)


        np.random.seed(43)
        np.random.shuffle(x_test)


        np.random.seed(43)
        np.random.shuffle(y_test)

        x_train = cv2.normalize(x_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_train = cv2.normalize(y_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        x_test = cv2.normalize(x_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_test = cv2.normalize(y_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        x_train = adiciona_a_dimensao_das_cores(x_train)
        y_train = adiciona_a_dimensao_das_cores(y_train)
        x_test = adiciona_a_dimensao_das_cores(x_test)
        y_test = adiciona_a_dimensao_das_cores(y_test)
    elif dataset.lower() == '25x25':
        TRIAL_DIRECTORY = "/content/gdrive/My Drive/Colab Notebooks/dataset/Phantoms/Alvarado/"
        path = os.path.join(TRIAL_DIRECTORY, "numpy", "25x25")

        x_train = np.load(os.path.join(path, 'x_train.npy'))
        y_train = np.load(os.path.join(path, 'y_train.npy'))
        x_test = x_train
        y_test = y_train

  
    if 'normalize' in config and config['normalize']:
        x_train = cv2.normalize(x_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_train = cv2.normalize(y_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        x_test = cv2.normalize(x_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_test = cv2.normalize(y_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

    if 'shuffle' in config and config['shuffle']:
        np.random.seed(13)
        np.random.shuffle(x_train)

        np.random.seed(13)
        np.random.shuffle(y_train)


        np.random.seed(43)
        np.random.shuffle(x_test)


        np.random.seed(43)
        np.random.shuffle(y_test)

    if 'divide_by_255' in config and config['divide_by_255']:
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

    (x_train, y_train, _, _) = load_data(check, config)

    mlp = MLP(image_dimension=IMAGE_DIMENSION, hidden_layers=3, depth=32, multiply=True)
    mlp.compile(optimizer="adam", learning_rate=0.001, loss="mse")

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

    (x_train, y_train, _, _) = load_data(check, config)

    cnn = CNN(image_dimension=IMAGE_DIMENSION, hidden_layers=10, depth=32, multiply=False, pooling=None)
    cnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')
    
    ckpt = os.path.join(config['metadata_path'], f'CNN.hdf5')
    cnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    cnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)
    time_in_seconds = time.time() - start_time

    cnn.save_loss_plot(os.path.join(config['output_path'], f'cnn_loss.png'))

    append_in_csv(config['time_file'], f'cnn, {time_in_seconds}')

def train_autoencoder(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.denoising_autoencoder import DenoisingAutoencoder

    (x_train, y_train, _, _) = load_data(check, config)

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

    (x_train, y_train, _, _) = load_data(check, config)
    
    gan = CGanDenoiser(image_dimensions=IMAGE_DIMENSION)
    gan.compile(optimizer='adam', learning_rate=0.001, loss='mse')
    gan.set_checkpoint(directory=os.path.join(config['metadata_path'], 'cgan-ckpt'))

    gan.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=256)
    time_in_seconds = time.time() - start_time

    append_in_csv(config['time_file'], f'gan, {time_in_seconds}')


def train_dncnn(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    (x_train, y_train, _, _) = load_data(check, config)

    dncnn = DnCNN(number_of_layers=19)
    dncnn.compile(optimizer="adam", learning_rate=0.0001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'DnCNN.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)
    dncnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN_loss.png'))

    append_in_csv(config['time_file'], f'dncnn, {time_in_seconds}')


def train_dncnn_10(file: str, check: bool, config: Dict[str, str]):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    (x_train, y_train, _, _) = load_data(check, config)

    dncnn = DnCNN(number_of_layers=10)
    dncnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    ckpt = os.path.join(config['metadata_path'], f'DnCNN10.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)
    dncnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN10_loss.png'))

    append_in_csv(config['time_file'], f'dncnn10, {time_in_seconds}')

def train(filename: str, check: bool):
    config = load_config(filename)

    config['time_file'] = os.path.join(config['metadata_path'], 'time.csv')
    append_in_csv(config['time_file'], 'name, seconds') 

    # removido o autoencoder
    for func in [train_mlp, train_cnn, train_cgan, train_dncnn, train_dncnn_10]:
        p = Process(target=func, args=(filename, check, config, ))
        p.start()
        p.join()
