import os
import numpy as np

import matplotlib.pyplot as plt

from denoising.methods.neural_network import NeuralNetwork

class CNN(NeuralNetwork):
    def __init__(self, image_dimension=(50,50), hidden_layers=3, depth=32, multiply=False, kernel_size=(3,3), pooling='maxpooling'):
        super().__init__()
        
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import Session

        config = ConfigProto(device_count = {'GPU': 0})
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        session = Session(config=config)
        del os.environ['CUDA_VISIBLE_DEVICES']
        
        self.hidden_layers = hidden_layers
        self.image_dimension = image_dimension
        self.depth = depth
        self.multiply = multiply
        self.pooling = pooling
        self.kernel_size = kernel_size

        self.build()

    def build(self):
        import keras
        from keras import Sequential, layers, activations
        from keras.models import Model

        self.input_shape = (self.image_dimension[0], self.image_dimension[1], 1)
        
        model = keras.Sequential()
        input = layers.Input(shape=self.input_shape, name='input')

        output = layers.Conv2D(self.depth, kernel_size=self.kernel_size, activation="relu", name="conv2d0")(input)
        if self.pooling == 'maxpooling':
            output = layers.MaxPooling2D(pool_size=(2,2))(output)
        
        for layer in range(1, self.hidden_layers):
            self.depth *= 2 if self.multiply else 1

            output = layers.Conv2D(self.depth, kernel_size=self.kernel_size, activation="relu", name=f"conv2d{layer}")(output)
            if self.pooling == 'maxpooling':
                output = layers.MaxPooling2D(pool_size=(2,2))(output)

        output = layers.Flatten()(output)
        output = layers.Dense(self.input_shape[0] * self.input_shape[1])(output)
        output = layers.Dropout(0.5)(output)
        output = layers.Reshape(self.input_shape, name='output')(output)

        self.model = Model(inputs=input, outputs=output)
