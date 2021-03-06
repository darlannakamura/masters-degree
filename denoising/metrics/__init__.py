import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from denoising.utils import *
from typing import List

def _validate_inputs(array1: np.ndarray, array2: np.ndarray):
    validate_array_input(array1)
    validate_array_input(array2)

    dimensions = len(array1.shape)
    
    for dimension in range(dimensions):
        assert array1.shape[dimension] == array2.shape[dimension], \
            f'The inputs arrays should have the same dimensions. Received: {array1.shape} and {array2.shape}'

def _need_to_normalize(array: np.ndarray) -> bool:
    if array.min() == 0 and array.max() == 255:
        return False
    
    return True

def psnr(ground_truth: np.ndarray, noisy: np.ndarray) -> np.ndarray:
    """Calculate the PSNR of every image of the dataset.
    We expect that ground_truth and noisy have the same dimensions. 

    Args:
        ground_truth (np.ndarray): expecting a np.ndarray of four dimensions.
        noisy (np.ndarray): expecting a np.ndarray of four dimensions. 
        
        The ground_truth and noisy should have the exactly same dimensions. 

    Returns:
        np.ndarray: return the PSNR of every image. To extract the mean, just use .mean().
    """
    _validate_inputs(ground_truth, noisy)
    assert np.issubdtype(ground_truth.dtype, np.integer) == True, "ground_truth should have dtype as integer, e.g: uint8."
    assert np.issubdtype(noisy.dtype, np.integer) == True, "noisy should have dtype as integer, e.g: uint8"
    
    psnr_acumulated = []
    images_quantity = ground_truth.shape[0]

    for i in range(images_quantity):
        psnr_image = peak_signal_noise_ratio(
            ground_truth[i,:,:,0], 
            noisy[i,:,:,0],
            data_range=255
        )
        psnr_acumulated.append(psnr_image)
    
    return np.array(psnr_acumulated)

def ssim(ground_truth: np.ndarray, noisy: np.ndarray) -> List[float]:
    """Calculate the SSIM of every image of the dataset.
    We expect that ground_truth and noisy have the same dimensions. 

    Args:
        ground_truth (np.ndarray): expecting a np.ndarray of four dimensions.
        noisy (np.ndarray): expecting a np.ndarray of four dimensions. 
        
        The ground_truth and noisy should have the exactly same dimensions. 

    Returns:
        np.ndarray: return the SSIM of every image. To extract the mean, just use .mean().
    """
    _validate_inputs(ground_truth, noisy)
    
    assert np.issubdtype(ground_truth.dtype, np.integer) == True, "ground_truth should have dtype as integer, e.g: uint8."
    assert np.issubdtype(noisy.dtype, np.integer) == True, "noisy should have dtype as integer, e.g: uint8"
    
    ssim_accumulated = []
    images_quantity = ground_truth.shape[0] 
    
    for i in range(images_quantity):
        ssim_image = structural_similarity(
            ground_truth[i,:,:,0], 
            noisy[i,:,:,0],
            data_range=255
        )
        ssim_accumulated.append(ssim_image)
    
    return np.array(ssim_accumulated)
