import os
import numpy as np
from denoising.utils import *
from typing import Callable

from tqdm import tqdm

from multiprocessing import Pool

def parallel(noisy_images: np.ndarray, function: Callable, noise_std_dev: float, num_threads:int=None, **kwargs) -> np.ndarray:
    if num_threads is None:
        num_threads = os.cpu_count()

    params = []

    total = noisy_images.shape[0]

    quantity_per_thread = total // num_threads

    for i in range(0, total, quantity_per_thread):
        params.append({
            'noisy_images': noisy_images[:, i:i+quantity_per_thread , :, :],
            'noise_std_dev': noise_std_dev,
            **kwargs
        })

    outputs = None
    with Pool(num_threads) as pool:
        # output has the output of each thread
        outputs = pool.map(function, params)
    
    array = outputs[0]
    
    for output in outputs[1:]:
        array = np.concatenate((array, output), axis=0)
    
    return array
    
def _batch_algorithm_implementation(single_image_denoising_algorithm, noisy_images: np.ndarray, noise_std_dev: float, show_progress:bool = False, *args, **kwargs):
    validate_array_input(noisy_images)
    validate_if_noise_std_dev_is_a_float(noise_std_dev)

    filtered_images = []

    if show_progress:
        for i in tqdm(range(noisy_images.shape[0])):
            filtered_images.append(
                single_image_denoising_algorithm(
                    noisy_images[i, :,:,:], 
                    *args,
                    **kwargs
                )
            )
    else:
        for i in range(noisy_images.shape[0]):
            filtered_images.append(
                single_image_denoising_algorithm(
                    noisy_images[i, :,:,:], 
                    *args,
                    **kwargs
                )
            )
    
    filtered_images = np.array(filtered_images)

    return filtered_images
