import os
import numpy as np

import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, run_in_cpu=False):
        self.has_checkpoint = False

        if run_in_cpu:
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import Session

            config = ConfigProto(device_count = {'GPU': 0})
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

            session = Session(config=config)
            del os.environ['CUDA_VISIBLE_DEVICES']

    def compile(self, optimizer: str, learning_rate: float, loss: str):
        import tensorflow as tf
        import keras
        from keras.models import load_model

        AVAILABLE_OPTIMIZERS = ['adam']
        AVAILABLE_LOSS = ['mse']

        assert optimizer in AVAILABLE_OPTIMIZERS, f'Available optimizers are: {AVAILABLE_OPTIMIZERS}'
        assert loss in AVAILABLE_LOSS, f'Available loss are: {AVAILABLE_LOSS}'

        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate)
        
        self.model.compile(optimizer=opt, loss=loss)

    def set_checkpoint(self, filename, save_best_only=True, save_weights_only=False):
        import keras
        self.has_checkpoint = True

        self.checkpoint = keras.callbacks.ModelCheckpoint(
            filename, 
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            monitor="val_loss",
            save_freq="epoch"
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int,
        batch_size: int, shuffle: bool = False, extract_validation_dataset: bool=True):
        kw = {
            'epochs': epochs,
            'batch_size': batch_size,
            'shuffle': shuffle
        }

        if extract_validation_dataset:
            total = x_train.shape[0]

            x_val, y_val, x_train, y_train = (
                x_train[:int(total*0.1)], 
                y_train[:int(total*0.1)],
                x_train[int(total*0.1):],
                y_train[int(total*0.1):]
            )

            kw["validation_data"] = (x_val, y_val)

        if self.has_checkpoint:
            kw["callbacks"] = [self.checkpoint]

        self.history = self.model.fit(
            x_train,
            y_train,
            **kw
        )

    def summary(self):
        self.model.summary()

    def save_loss_plot(self, filename: str):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.legend(['loss', 'val. loss'])
        plt.savefig(filename)
        plt.close()

    def load(self, filename):
        self.model.load_weights(filename)

    def test(self, x_test: np.ndarray, verbose: int = 2):
        return self.model.predict(x_test, verbose=verbose)
