import tensorflow as tf
import numpy as np
import keras
from keras import Sequential, layers, activations
from keras.models import Model

import matplotlib.pyplot as plt

from denoising.methods.neural_network import NeuralNetwork

class MLP(NeuralNetwork):
    def __init__(self, image_dimension=(50,50), hidden_layers=3, depth=32, multiply=False):
        super().__init__()
        
        self.hidden_layers = hidden_layers
        self.image_dimension = image_dimension
        self.depth = depth
        self.multiply = multiply

        self.build()

    def build(self):
        self.input_shape = (self.image_dimension[0], self.image_dimension[1], 1)
        
        model = keras.Sequential()
        input = layers.Input(shape=self.input_shape, name='input')

        output = layers.Dense(self.depth, activation="relu", name=f"dense{layer}")(input)

        for layer in range(1, self.hidden_layers):
            if multiply:
                self.depth = self.depth * 2

            output = layers.Dense(self.depth, activation="relu", name=f"dense{layer}")(output)
        
        output = layers.Flatten()(output)
        output = layers.Reshape(self.input_shape)(output)

        self.model = Model(inputs=input, outputs=output)

