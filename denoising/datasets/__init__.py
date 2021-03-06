import os
import cv2
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from denoising.datasets.dicom_processing import get_nparray_from_dicom
from skimage.util import random_noise
from denoising.utils import normalize

class DirectoryNotFoundError(Exception):
    def __init__(self, directory: str):
        super().__init__(f"The directory '{directory}' was not found. Please, check if this directory really exists.")

class LoadDatasetError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

def get_files_in_directory(path: str, extension: str = '.dcm') -> List[str]:
    """Return a list of string with all file of a certain extension.
    By default, with dicom extension.

    Args:
        path (str): path to search files
        extension (str): extension of the files. Defaults to '.dcm'.

    Returns:
        List[str]: list of file paths.
    """
    result = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            result += get_files_in_directory(os.path.join(path, d), extension)
        
        for f in files:
            if f.endswith(extension):
                result.append(os.path.join(path, f))
        
        return result

def get_projections(directory_path: str, quantity: int = 15) -> np.ndarray:
    # path = datasets[phantom][dataset_name]
    projections = []
    
    files = get_files_in_directory(directory_path, '.dcm')
    if files:
        files = files[:quantity]
    
    for file in files:
        projections.append(get_nparray_from_dicom(file))
    
    return np.array(projections)

def extract_patches(np_array: np.ndarray, begin: Tuple[int,int] = (80,934), \
    stride: int = 10, dimension: Tuple[int,int] = (50,50), \
    quantity_per_image: Tuple[int, int] = (5,2)) -> np.ndarray:

    assert type(np_array) == np.ndarray, "np_array should be a np.ndarray."
    assert len(np_array.shape) == 4, "np_array should have four dimensions."

    assert type(begin) == tuple
    assert len(begin) == 2

    assert type(stride) == int

    assert type(dimension) == tuple
    assert len(dimension) == 2

    assert type(quantity_per_image) == tuple
    assert len(quantity_per_image) == 2

    patches = []
    for img_index in range(0, np_array.shape[0]):
        
        for i in range(0, quantity_per_image[0]*stride, stride):
            for j in range(0, quantity_per_image[1]*stride, stride):
                p = np_array[img_index, (begin[0]+i):(begin[0]+dimension[0]+i), (begin[1]+j):(begin[1]+dimension[1]+j)]
                patches.append(p)
    
    return np.array(patches)
    
# TODO - extract_patches_from_mama_region
# usar um alg. de segmentação pra extrair só da região da mama.

def load_bsd300(dir_path: str) -> np.ndarray:
    """Load Berkeley Segmentation Dataset 300 from an specific dir_path,
    with all '.jpg' photos.

    It will return a np.array of four dimensions: (images, dimX, dimY, 1).
    This last dimensions is the grayscale dimension.
    Currently, this method don't support loading colored images.

    Args:
        dir_path (str): path for BSD300 .jpg images.

    Returns:
        np.ndarray: returns a 4 dimension np.ndarray: (300, 321, 481. 1).
    """
    if not os.path.isdir(dir_path):
        raise DirectoryNotFoundError(dir_path)

    files = get_files_in_directory(dir_path, extension='.jpg')

    if len(files) == 0:
        raise LoadDatasetError(f"None '.jpg' file was found in: '{dir_path}'.")

    arr = []

    for f in files:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    
        if img.shape[1] == 321:
            img = np.rot90(img)
    
        arr.append(img)
    
    nparr = np.array(arr)
    return nparr.reshape(nparr.shape + (1,))

def load_dataset(np_array: np.ndarray, shuffle:bool=False, split:Tuple[int, int]=(80,20)) -> Tuple[np.ndarray, np.ndarray]:
    """Load a train and test dataset, from a np_array of 4 dimensions.

    Args:
        np_array (np.ndarray): should have 4 dimensions.
        shuffle (bool, optional): if should shuffle before split. 
            Defaults to False, but we strongly recommend to shuffle, to reduce the bias in both datasets.
        split (Tuple[int, int], optional): percentage of train and test set, respectively. Defaults to (80,20).

    Returns:
        Tuple[np.ndarray, np.ndarray]: return two nparrays, the train and test set, respectively.
    """
    assert type(np_array) == np.ndarray, "np_array should be a np.ndarray."
    assert len(np_array.shape) == 4, "np_array should have four dimensions."

    assert type(split) == tuple, "split should be a tuple."
    assert len(split) == 2, "split should be a tuple with two dimensions, being the first term the percentage of train and the secon the percentage of test"
    assert split[0] + split[1] == 100, "split should be the percentage of train and test set, respectively, totalizing 100%."

    np_array_copy = deepcopy(np_array)

    if shuffle:
        np.random.shuffle(np_array_copy)

    p_train, p_test = split

    total = np_array.shape[0]
    train_percentage = p_train/100
    test_percentage = p_test/100
    
    return (np_array_copy[:int(train_percentage*total)], np_array_copy[int(train_percentage*total):])

def add_noise(np_array: np.ndarray, noise:str='gaussian', mean:float=None, var:float=None) -> np.ndarray:
    assert type(np_array) == np.ndarray, "np_array should be a np.ndarray."
    assert len(np_array.shape) == 4, "np_array should have four dimensions."

    assert noise in ('gaussian', 'poisson'), "noise should be 'gaussian' or 'poisson'"

    if noise == 'gaussian':
        assert mean != None, "mean should be defined when the noise is gaussian"
        assert var != None, "var should be defined when the noise is gaussian"
        assert type(mean) == float, "mean should be float"
        assert type(var) == float, "var should be float"

    noisy = []
    img_dimensions = (np_array[1], np_array[2])

    for i in range(np_array.shape[0]):
        img = np_array[i]

        if noise == 'gaussian':
            # noisy.append(random_noise(img, mode='gaussian', mean=mean, var=var))
            std = var ** 0.5
            noise = np.random.normal(mean, std, img.shape)
            noisy.append(img + noise)
        else:
            noisy.append(np.random.poisson(lam=img, size=None))

    nparr =  np.array(noisy)
    # return normalize(nparr, data_type='int')
    return nparr
