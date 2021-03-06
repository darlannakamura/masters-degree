import os
from typing import Tuple
import numpy as np

from denoising.methods.neural_network import NeuralNetwork

class DenoisingAutoencoder(NeuralNetwork):
    """Adapted from: 
    https://www.pyimagesearch.com/2020/02/24/denoising-autoencoders-with-keras-tensorflow-and-deep-learning/
    """
    def __init__(self, image_dimension: Tuple[int, int] = (50,50), \
        filters: Tuple[int, int] = (32, 64), latent_dimension: int = 16, run_in_cpu=False):
        super().__init__(run_in_cpu)

        self.width = image_dimension[0]
        self.height = image_dimension[1]

        self.filters = filters
        self.latent_dimension = latent_dimension

        self.build()

    def build(self):
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.layers import Conv2D
        from tensorflow.keras.layers import Conv2DTranspose
        from tensorflow.keras.layers import LeakyReLU
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Flatten
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Reshape
        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        from tensorflow.keras import backend as K

        input_shape = (self.height, self.width, 1)
        channel_dimension = -1

        inputs = Input(shape=input_shape)
        x = inputs

        for f in self.filters:
            # apply a CONV => RELU => BN operation
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=channel_dimension)(x)
        
        volume_size = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(self.latent_dimension)(x)
        # build the encoder model
        encoder = Model(inputs, latent, name="encoder")

        # start building the decoder model which will accept the
        # output of the encoder as its inputs
        latent_inputs = Input(shape=(self.latent_dimension,))
        x = Dense(np.prod(volume_size[1:]))(latent_inputs)
        x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        # loop over our number of filters again, but this time in
        # reverse order
        for f in self.filters[::-1]:
            # apply a CONV_TRANSPOSE => RELU => BN operation
            x = Conv2DTranspose(f, (3, 3), strides=2,
                padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=channel_dimension)(x)
        # apply a single CONV_TRANSPOSE layer used to recover the
        # original depth of the image
        x = Conv2DTranspose(1, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)
        # build the decoder model
        decoder = Model(latent_inputs, outputs, name="decoder")
        # our autoencoder is the encoder + decoder
        autoencoder = Model(inputs, decoder(encoder(inputs)),
            name="autoencoder")
        
        self.encoder = encoder
        self.decoder = decoder
        self.model = autoencoder
