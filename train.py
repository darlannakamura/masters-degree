import os
import time
import yaml
import numpy as np
import math
import argparse
from multiprocessing import Process
from loaders.config_loader import load_config
from loaders.data_loader import DataLoader

from sklearn.model_selection import KFold

from typing import Tuple, Dict

EPOCHS = 40

def append_in_csv(filename: str, row: str):
    with open(filename, 'a') as f:
        f.write(row+'\n')


def get_epochs(check):
    return EPOCHS if not check else 1


def train_mlp(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.mlp import MLP

    mlp = MLP(image_dimension=image_dimension, hidden_layers=5, depth=32, multiply=True)
    mlp.compile(optimizer="adam", learning_rate=0.001, loss="mse")

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')

    ckpt = os.path.join(config['metadata_path'], f'MLP.hdf5')
    mlp.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    mlp.fit(epochs=get_epochs(check), batch_size=128, shuffle=True, 
        x_train=x_train, y_train=y_train,
        extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time

    mlp.save_loss_plot(os.path.join(config['output_path'], f'mlp_loss.png'))
    
    append_in_csv(config['time_file'], f'{iteration},mlp,{time_in_seconds}')


def train_cnn(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.cnn import CNN

    cnn = CNN(image_dimension=image_dimension, hidden_layers=5, depth=32, multiply=False, pooling=None)
    cnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')
    
    ckpt = os.path.join(config['metadata_path'], f'CNN.hdf5')
    cnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)

    cnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)
    time_in_seconds = time.time() - start_time

    cnn.save_loss_plot(os.path.join(config['output_path'], f'cnn_loss.png'))

    append_in_csv(config['time_file'], f'{iteration},cnn,{time_in_seconds}')

def train_autoencoder(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.denoising_autoencoder import DenoisingAutoencoder

    ae = DenoisingAutoencoder(image_dimension=image_dimension)
    ae.compile(optimizer='adam', learning_rate=1e-3, loss='mse')

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')
    
    ckpt = os.path.join(config['metadata_path'], f'Autoencoder.hdf5')
    ae.set_checkpoint(ckpt, save_best_only=True, save_weights_only=False)

    ae.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    ae.save_loss_plot(os.path.join(config['output_path'], f'Autoencoder_loss.png'))

    append_in_csv(config['time_file'], f'{iteration},autoencoder,{time_in_seconds}')


def train_cgan(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.cgan_denoiser.main import CGanDenoiser
    
    gan = CGanDenoiser(image_dimensions=image_dimension)
    gan.compile(optimizer='adam', learning_rate=0.001, loss='mse')

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')
    
    gan.set_checkpoint(directory=os.path.join(config['metadata_path'], 'cgan-ckpt'))

    gan.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=256)
    time_in_seconds = time.time() - start_time

    append_in_csv(config['time_file'], f'{iteration},gan,{time_in_seconds}')


def train_dncnn(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    dncnn = DnCNN(number_of_layers=19)
    dncnn.compile(optimizer="adam", learning_rate=0.0001, loss='mse')

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')
    
    ckpt = os.path.join(config['metadata_path'], f'DnCNN.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)
    dncnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN_loss.png'))

    append_in_csv(config['time_file'], f'{iteration},dncnn,{time_in_seconds}')


def train_dncnn_10(check: bool, config: Dict[str, str], x_train, y_train, image_dimension, iteration):
    start_time = time.time()
    from denoising.methods.neural_network.dncnn import DnCNN

    dncnn = DnCNN(number_of_layers=10)
    dncnn.compile(optimizer="adam", learning_rate=0.001, loss='mse')

    config['output_path'] = os.path.join(config['output_path'], f'{iteration}')
    config['metadata_path'] = os.path.join(config['output_path'], '.metadata')
    
    ckpt = os.path.join(config['metadata_path'], f'DnCNN10.hdf5')
    dncnn.set_checkpoint(filename=ckpt, save_best_only=True, save_weights_only=False)
    dncnn.fit(epochs=get_epochs(check), x_train=x_train, y_train=y_train, batch_size=128, shuffle=True, extract_validation_dataset=True)

    time_in_seconds = time.time() - start_time
    dncnn.save_loss_plot(os.path.join(config['output_path'], f'DnCNN10_loss.png'))

    append_in_csv(config['time_file'], f'{iteration},dncnn10, {time_in_seconds}')

def train(filename: str, check: bool):
    config = load_config(filename)

    config['time_file'] = os.path.join(config['output_path'], 'time.csv')
    append_in_csv(config['time_file'], 'iteration,name,seconds')

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    data_loader = DataLoader(config=config, check=check)
    image_dimension = data_loader.get_patch_dimension()

    x_train, y_train, x_test, y_test = data_loader.get()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    iteration = 0

    for train_index, _ in kfold.split(x,y):
        x_train, y_train = x[train_index], y[train_index]
   
        for func in [train_mlp, train_cnn, train_autoencoder, train_cgan, train_dncnn, train_dncnn_10]:
            p = Process(target=func, args=(check, config, x_train, y_train, image_dimension, iteration, ))
            p.start()
            p.join()

        iteration += 1