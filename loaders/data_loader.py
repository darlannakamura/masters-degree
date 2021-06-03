import os, cv2, math
from settings import BSD300_DIR, PROJECTIONS_DIR
import numpy as np
from typing import Dict, Tuple

from denoising.datasets import load_bsd300
from denoising.datasets import load_dataset
from denoising.datasets import extract_patches
from denoising.datasets import add_noise

DEFAULT_DIMENSION = (50,50)

class DataLoader:
    def __init__(self, config: Dict[str, str], check = False):
        self.config = config
        self.is_testing = check

        self.x_train, self.y_train, self.x_test, self.y_test = self.get_dataset(config['dataset'])
        self.set_noise_statistics()

    def get_train(self):
        return (self.x_train, self.y_train)

    def get_test(self):
        return (self.x_test, self.y_test)

    def get(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def get_dataset(self, dataset_name: str):
        (train_size, test_size) = self.get_size()

        if dataset_name.lower() == 'bsd300':
            y_train, y_test = self.load_bsd300()
            x_train, x_test = self.add_noise(y_train, y_test)

        elif dataset_name.lower() == 'dbt':
            x_train, y_train, x_test, y_test = self.load_dbt()
        elif dataset_name.lower() == 'spie_2021':
            x_train, y_train, x_test, y_test = self.load_spie_2021()
        elif dataset_name.lower() == '25x25':
            x_train, y_train, x_test, y_test = self.load_25x25()

        if self.should_normalize():
            x_train = cv2.normalize(x_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            y_train = cv2.normalize(y_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            x_test = cv2.normalize(x_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            y_test = cv2.normalize(y_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        if self.should_shuffle():
            np.random.seed(13)
            np.random.shuffle(x_train)
            np.random.seed(13)
            np.random.shuffle(y_train)
            np.random.seed(43)
            np.random.shuffle(x_test)
            np.random.seed(43)
            np.random.shuffle(y_test)

        if self.should_divide_by_255():
            x_train = x_train.astype('float32') / 255.0
            y_train = y_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
            y_test = y_test.astype('float32') / 255.0

        if self.is_testing:
            return (x_train[:10], y_train[:10], x_test[:10], y_test[:10])

        if train_size and test_size:
            return (x_train[:train_size], y_train[:train_size], \
                x_test[:test_size], y_test[:test_size])

        return (x_train, y_train, x_test, y_test)


    def set_noise_statistics(self):
        noise = self.config.get('noise', None)

        if not noise:
            noise = self.y_test - self.x_test
            self.mean = noise.mean()
            self.variance = np.var(noise)
            self.std = noise.std()

        if isinstance(noise, str) and noise == 'poisson':
            return 0.1
        
        if isinstance(noise, dict):
            assert 'type' in noise, "noise should have 'type' attribute. Options are: gaussian and poisson-gaussian."
            noise_type = noise['type']

            assert 'mean' in noise, "noise should have 'mean' attribute."
            assert 'variance' in noise, "noise should have 'variance' attribute."
            
            self.mean = float(noise['mean'])
            self.variance = float(noise['variance'])
            self.std = math.sqrt(self.variance)
        
    def get_noise_mean(self) -> float:
        return self.mean

    def get_noise_variance(self) -> float:
        return self.variance
    
    def get_noise_std(self) -> float:
        return self.std


    def load_bsd300(self) -> Tuple[np.ndarray, np.ndarray]:
        imgs = load_bsd300(BSD300_DIR)
        patch_dimension = self.get_patch_dimension()
        patches = extract_patches(imgs, begin=(0,0), stride=10,
            dimension=patch_dimension, quantity_per_image=(10,10))
        
        if self.should_shuffle():
            np.random.seed(10)
            np.random.shuffle(patches)

        y_train, y_test = load_dataset(patches, shuffle=False, split=(80,20))

        return y_train, y_test

    def load_dbt(self) -> Tuple[np.ndarray, np.ndarray]:
        noisy_projections = np.load(os.path.join(PROJECTIONS_DIR, 'noisy_10.npy'))
        noisy_projections = noisy_projections.reshape((-1, 1792, 2048, 1))

        if self.should_shuffle():
            np.random.seed(10)
            np.random.shuffle(noisy_projections)
        
        patch_dimension = self.get_patch_dimension()

        noisy_patches = extract_patches(noisy_projections, begin=(0,500), stride=10,
            dimension=patch_dimension, quantity_per_image=(10,10))

        x_train, x_test = load_dataset(noisy_patches, shuffle=False, split=(80,20))

        original_projections = np.load(os.path.join(PROJECTIONS_DIR, 'original_10.npy'))
        original_projections = original_projections.reshape((-1, 1792, 2048, 1))
        
        if self.should_shuffle():
            np.random.seed(10)
            np.random.shuffle(original_projections)
        
        original_patches = extract_patches(original_projections, begin=(0,500), stride=10,
            dimension=patch_dimension, quantity_per_image=(10,10))

        y_train, y_test = load_dataset(original_patches, shuffle=False, split=(80,20))

        return x_train, y_train, x_test, y_test

    def add_noise(self, y_train, y_test) -> Tuple[np.ndarray, np.ndarray]:
        noise = self.config.get('noise', None)

        if not noise:
            raise AssertionError("Config should have a noise type declared.")

        if isinstance(noise, str) and noise == 'poisson':
            x_train = add_noise(y_train, noise='poisson')
            x_test = add_noise(y_test, noise='poisson')
        if isinstance(noise, dict):
            assert 'type' in noise, "noise should have 'type' attribute. Options are: gaussian and poisson-gaussian."
            noise_type = noise['type']
            assert 'mean' in noise, "noise should have 'mean' attribute."
            assert 'variance' in noise, "noise should have 'variance' attribute."

            mean = float(noise['mean'])
            variance = float(noise['variance'])

            if noise_type == 'gaussian':
                x_train = add_noise(y_train, noise='gaussian', mean=mean, var=variance)
                x_test = add_noise(y_test, noise='gaussian', mean=mean, var=variance)
            elif noise_type == 'poisson-gaussian':
                x_train = add_noise(y_train, noise='poisson')
                x_test = add_noise(y_test, noise='poisson')

                x_train = add_noise(y_train, noise='gaussian', mean=mean, var=variance)
                x_test = add_noise(y_test, noise='gaussian', mean=mean, var=variance)

        return x_train, x_test

    def load_spie_2021(self):
        from denoising.datasets.spie_2021 import carrega_dataset, adiciona_a_dimensao_das_cores

        full_x_train, full_y_train, full_x_test, full_y_test = carrega_dataset(
          '/content/gdrive/My Drive/Colab Notebooks/dataset/patch-50x50-cada-projecao-200', 
          divisao=(80,20), embaralhar=True)

        x_train =  np.reshape(full_x_train, (-1, 50, 50))
        x_test = np.reshape(full_x_test, (-1, 50, 50))
        y_train = np.reshape(full_y_train, (-1, 50, 50))
        y_test = np.reshape(full_y_test, (-1, 50, 50)) 

        del full_x_train
        del full_y_train
        del full_x_test
        del full_y_test

        x_train = x_train[:15000]
        y_train = y_train[:15000]

        x_test = x_test[:3750]
        y_test = y_test[:3750]

        np.random.seed(13)
        np.random.shuffle(x_train)

        np.random.seed(13)
        np.random.shuffle(y_train)

        np.random.seed(43)
        np.random.shuffle(x_test)

        np.random.seed(43)
        np.random.shuffle(y_test)

        x_train = cv2.normalize(x_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_train = cv2.normalize(y_train, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        x_test = cv2.normalize(x_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        y_test = cv2.normalize(y_test, None, alpha= 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)

        x_train = adiciona_a_dimensao_das_cores(x_train)
        y_train = adiciona_a_dimensao_das_cores(y_train)
        x_test = adiciona_a_dimensao_das_cores(x_test)
        y_test = adiciona_a_dimensao_das_cores(y_test)


    def load_25x25(self):
        TRIAL_DIRECTORY = "/content/gdrive/My Drive/Colab Notebooks/dataset/Phantoms/Alvarado/"
        path = os.path.join(TRIAL_DIRECTORY, "numpy", "25x25")

        x_train = np.load(os.path.join(path, 'x_train.npy'))
        y_train = np.load(os.path.join(path, 'y_train.npy'))
        x_test = x_train
        y_test = y_train

        return x_train, y_train, x_test, y_test

    def get_patch_dimension(self) -> Tuple[int,int]:
        dim = self.config.get('dimension', None)

        if not dim:
            return DEFAULT_DIMENSION
        
        if 'x' in dim:
            dim = dim.split('x')
            return (int(dim[0]), int(dim[1]))
        else:
            raise AssertionError("dimension should be: 50x50 format.")

    def should_shuffle(self) -> bool:
        return self.config.get('shuffle', False)

    def should_normalize(self):
        return self.config.get('normalize', False)
    
    def should_divide_by_255(self):
        return self.config.get('divide_by_255', False)
    
    def get_size(self):
        size = self.config.get('size', None)

        if isinstance(size, dict):
            train_size = self.config['size'].get('train', None)
            test_size = self.config['size'].get('test', None)

            assert isinstance(train_size, int) and isinstance(test_size, int), \
                "Train and test size should be int."

            return (train_size, test_size)

        return (None, None)
