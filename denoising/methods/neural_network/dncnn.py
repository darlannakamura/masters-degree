import numpy as np
from keras import Sequential, layers, activations
from keras.models import Model

import matplotlib.pyplot as plt

from denoising.methods.neural_network import NeuralNetwork

class DnCNN(NeuralNetwork):
    def __init__(self, number_of_layers=19, run_in_cpu=False):
        super().__init__(run_in_cpu)
        
        self.number_of_layers = number_of_layers

        self.build()

    def build(self):
        model = Sequential()
        input = layers.Input(shape=(None, None, 1), name='input')

        output = layers.Conv2D(filters=64,kernel_size=(3,3), strides=(1,1), 
                            padding='same', name='conv1')(input)
        output = layers.Activation('relu')(output)

        for layer in range(2, self.number_of_layers):
            output = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), 
                                padding='same', name='conv%d' %layer)(output)
            output = layers.BatchNormalization(axis=-1, epsilon=1e-3, name='batch_normalization%d' %layer)(output)
            output = layers.Activation('relu')(output)

        output = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', strides=(1,1), name=f'conv{self.number_of_layers}')(output)

        output = layers.Subtract(name='subtract')([input, output])

        self.model = Model(inputs=input, outputs=output)

