import numpy as np

from denoising.methods.neural_network.deep_image_prior.main import DeepImagePrior
from denoising.utils import *


def deep_image_prior(noisy_images: np.ndarray, noise_std_dev=1.0, iterations=100) -> np.ndarray:    
    validate_array_input(noisy_images)

    filtered_images = []

    dip = DeepImagePrior()

    # convert (52,52,1) to (1,52,52)
    noisy_images = noisy_images[:, :, :, 0]
    noisy_images = noisy_images.reshape((1,) + noisy_images.shape)

    noisy_images = normalize(noisy_images, interval=(0,1), data_type='float')

    for i in range(noisy_images.shape[1]):
        img = noisy_images[:, i, :, :]
        predicted_image, predicted_image_avg = dip.run(
            iterations=iterations,
            noise_std_dev=noise_std_dev,
            image_noisy=img
        )

        predicted_image = predicted_image[0, :, :]

        filtered_images.append(predicted_image)
    
    filtered_images = np.array(filtered_images)
    return filtered_images.reshape(filtered_images.shape + (1,))
