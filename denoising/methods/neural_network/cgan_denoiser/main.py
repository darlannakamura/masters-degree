from __future__ import absolute_import, division, print_function

import os
import PIL
import time
import math
import glob
import imageio
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np


class CGanDenoiser:
    def __init__(self, image_dimensions=(52,52), run_in_cpu=False):
        import denoising.methods.neural_network.cgan_denoiser.cgan as model
        import tensorflow as tf

        assert image_dimensions[0] == image_dimensions[1], "Image dimension should be squared."

        if run_in_cpu:
            from tensorflow.compat.v1 import ConfigProto
            from tensorflow.compat.v1 import Session

            config = ConfigProto(device_count = {'GPU': 0})
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

            session = Session(config=config)
            del os.environ['CUDA_VISIBLE_DEVICES']

        # # Set up the models for training
        self.generator = model.make_generator_model_small(train_size=image_dimensions[0])
        self.discriminator = model.make_discriminator_model(train_size=image_dimensions[0]) 

        tf.compat.v1.enable_eager_execution()        


    def compile(self, optimizer: str, learning_rate: float, loss: str):
        import tensorflow as tf

        AVAILABLE_OPTIMIZERS = ['adam']
        assert optimizer == 'adam', f"Available optimizers are: {AVAILABLE_OPTIMIZERS}"
        
        self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    def set_checkpoint(self, directory: str = None, save_every=5):
        import tensorflow as tf

        if directory:
            self.checkpoint_prefix = directory

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

        self.save_every = save_every

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int):
        import tensorflow as tf

        train_inputs = x_train
        train_labels = y_train

        global_step = tf.compat.v1.train.get_or_create_global_step()

        buffer_size = train_inputs.shape[0]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))\
        .shuffle(buffer_size).batch(batch_size)

        # self.generator = model.make_generator_model_small(train_size=self.image_dimensions[0])
        # self.discriminator = model.make_discriminator_model(train_size=self.image_dimensions[0])

        print("\nTraining...\n")
        # Compile training function into a callable TensorFlow graph (speeds up execution)
        train_step = tf.function(self.train_step)
        self.train(train_dataset, epochs)
        print("\nTraining done\n")

    def train(self, dataset, epochs: int):
        import tensorflow as tf

        for epoch in range(epochs):
            start = time.time()

            for x, y in tf.compat.v1.data.make_one_shot_iterator(dataset):
                self.train_step(x, y)
            
            # saving (checkpoint) the model every few epochs
            if (epoch + 1) % self.save_every == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
            
            print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    
    def load(self, filename:str):
        self.checkpoint.restore(filename)
    
    def test(self, x_test: np.ndarray):
        prediction =  self.generator(x_test, training=False)
        return prediction.numpy()

    def train_step(self, inputs: np.ndarray, labels: np.ndarray):
        import tensorflow as tf
        from denoising.methods.neural_network.cgan_denoiser.data_processing import rmse, psnr 

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(inputs, training=True)

            real_output = self.discriminator(labels, training=True)
            generated_output = self.discriminator(generated_images, training=True)
                
            gen_d_loss = self.generator_d_loss(generated_output)
            gen_abs_loss = self.generator_abs_loss(labels, generated_images)
            gen_loss = gen_d_loss + gen_abs_loss
            gen_rmse = rmse(labels, generated_images)
            gen_psnr = psnr(labels, generated_images)
            disc_loss = self.discriminator_loss(real_output, generated_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.variables))

    def generator_d_loss(self, generated_output):
        import tensorflow as tf

        # [1,1,...,1] with generated images since we want the discriminator to judge them as real
        return tf.compat.v1.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


    def generator_abs_loss(self, labels, generated_images):
        import tensorflow as tf
        from denoising.methods.neural_network.cgan_denoiser.config import ConfigCGAN as config

        # As well as "fooling" the discriminator, we want particular pressure on ground-truth accuracy
        return config.L1_lambda * tf.compat.v1.losses.absolute_difference(labels, generated_images)  # mean


    def discriminator_loss(self, real_output, generated_output):
        import tensorflow as tf
        # [1,1,...,1] with real output since we want our generated examples to look like it
        real_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.ones_like(real_output), logits=real_output)

        # [0,0,...,0] with generated images since they are fake
        generated_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

        total_loss = real_loss + generated_loss

        return total_loss
