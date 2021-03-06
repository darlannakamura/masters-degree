import numpy as np
from sparselandtools.applications.denoising import KSVDImageDenoiser
from sparselandtools.pursuits import MatchingPursuit
from sparselandtools.dictionaries import DCTDictionary

from denoising.utils import *

def KSVD(noisy_images: np.ndarray, noise_std_dev: float) -> np.ndarray:
    """
    Params:
    noisy_images: receive noisy_images with shape (IMG, M, N, C), 
    where IMG is the quantity of noisy images 
    with dimensions MxN and
    C represents the color dimensions, i.e. if the images are colored,
    C = 3 (RGB), otherwise if C = 1, is grayscale. 

    noise_std_dev: the standart deviation from noise.
    """
    validate_array_input(noisy_images)
    validate_if_noise_std_dev_is_a_float(noise_std_dev)

    filtered_images = []

    # set patch size
    patch_size = 5

    # initialize denoiser
    initial_dictionary = DCTDictionary(patch_size, 11)
    denoiser = KSVDImageDenoiser(initial_dictionary, pursuit=MatchingPursuit)

    # denoise images
    for i in range(noisy_images.shape[0]):
        predicted_image, d, a = denoiser.denoise(noisy_images[i, :, :, 0], 
                sigma=noise_std_dev, 
                patch_size=patch_size
                # n_iter=4
            )

        filtered_images.append(predicted_image)
    
    filtered_images = np.array(filtered_images)
    return filtered_images.reshape(filtered_images.shape + (1,))
