from experiments.decorator import method

from denoising.methods.traditional.bm3d import BM3D
from denoising.methods.traditional.non_local_means import NLM
from denoising.methods.traditional.wavelet import wavelet_soft_thresholding 
from denoising.methods.traditional.wiener import wiener_filter
from denoising.methods.traditional.ksvd import KSVD

from denoising.methods.neural_network.dncnn import DnCNN
from denoising.methods.neural_network.denoising_autoencoder import DenoisingAutoencoder
from denoising.methods.neural_network.deep_image_prior import deep_image_prior
from denoising.methods.neural_network.mlp import MLP
from denoising.methods.neural_network.cnn import CNN
from denoising.methods.neural_network.cgan_denoiser.main import CGanDenoiser 

EPOCHS = 40

class Methods:
    @method
    def pwf():
        return {
            'instance': wiener_filter,
            'name': 'Wiener Filter'
        }

    @method
    def wst():
        return {
            'instance': wavelet_soft_thresholding,
            'name': 'WST'
        }

    @method
    def nlm():
        return {
            'instance': NLM,
            'name': 'NLM'
        }

    @method
    def bm3d():
        return {
            'instance': BM3D,
            'name': 'BM3D'
        }

    @method
    def ksvd():
        return {
            'instance': KSVD,
            'name': 'K-SVD'
        }

    @method
    def dip():
        return {
            'instance': deep_image_prior,
            'name': 'DIP'
        }

    @method
    def dncnn():
        name = "DnCNN"
        instance = DnCNN
        need_train = True
        parameters = {
            "__init__": {
                'number_of_layers': 19,
                "run_in_cpu": True
            },
            "compile": {
                "optimizer": "adam",
                "learning_rate": 0.0001,
                "loss": "mse"
            },
            "fit": {
                "epochs": EPOCHS,
                "batch_size": 128,
                "shuffle": True,
                "extract_validation_dataset": True
            },
            "set_checkpoint": {
                "filename": "default",
                "save_best_only": True,
                "save_weights_only": False
            }
        }

        return {
            'name': name,
            'instance': instance,
            'need_train': need_train,
            'parameters': parameters
        }

    @method
    def dncnn10():
        name = "DnCNN10"
        instance = DnCNN
        need_train = True
        parameters = {
            "__init__": {
                'number_of_layers': 10,
                "run_in_cpu": True
            },
            "compile": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "loss": "mse"
            },
            "fit": {
                "epochs": EPOCHS,
                "batch_size": 128,
                "shuffle": True,
                "extract_validation_dataset": True
            },
            "set_checkpoint": {
                "filename": "default",
                "save_best_only": True,
                "save_weights_only": False
            }
        }

        return {
            'name': name,
            'instance': instance,
            'need_train': need_train,
            'parameters': parameters
        }
    
    # @method
    def denoising_autoencoder():
        return {
            'name': 'Autoencoder',
            'instance': DenoisingAutoencoder,
            'need_train': True,
            "parameters": {
                "__init__": {
                    "image_dimension": (52,52),
                    "run_in_cpu": True
                },
                "compile": {
                    "optimizer": "adam",
                    "learning_rate": 1e-3,
                    "loss": "mse"
                },
                "fit": {
                    "epochs": EPOCHS,
                    "batch_size": 128,
                    "shuffle": True,
                    "extract_validation_dataset": True
                },
                "set_checkpoint": {
                    "filename": "default",
                    "save_best_only": True,
                    "save_weights_only": False
                }
            },
        }

    @method
    def multi_layer_perceptron():
        return {
            'name': 'MLP',
            'instance': MLP,
            'need_train': True,
            "parameters": {
                "__init__": {
                    "image_dimension": (50,50),
                    "hidden_layers": 3,
                    "depth": 32,
                    "multiply": True,
                    "run_in_cpu": True
                },
                "compile": {
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "loss": "mse"
                },
                "fit": {
                    "epochs": EPOCHS,
                    "batch_size": 128,
                    "shuffle": True,
                    "extract_validation_dataset": True
                },
                "set_checkpoint": {
                    "filename": "default",
                    "save_best_only": True,
                    "save_weights_only": False
                }
            },
        }

    @method
    def cnn():
        return {
            'name': 'CNN',
            'instance': CNN,
            'need_train': True,
            "parameters": {
                "__init__": {
                    "image_dimension": (50,50),
                    "hidden_layers": 10,
                    "depth": 32,
                    "multiply": False,
                    "pooling": None,
                    "run_in_cpu": True
                },
                "compile": {
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "loss": "mse"
                },
                "fit": {
                    "epochs": EPOCHS,
                    "batch_size": 128,
                    "shuffle": True,
                    "extract_validation_dataset": True
                },
                "set_checkpoint": {
                    "filename": "default",
                    "save_best_only": True,
                    "save_weights_only": False
                }
            },
        }

    @method
    def cgan_denoiser():
        return {
            'name': 'CGAN',
            'instance': CGanDenoiser,
            'need_train': True,
            "parameters": {
                "__init__": {
                    "image_dimensions": (50,50),
                    "run_in_cpu": True,
                },
                "compile": {
                    "optimizer": "adam",
                    "learning_rate": 0.0001,
                    "loss": "mse",
                },
                "fit": {
                    "epochs": EPOCHS,
                    "batch_size": 256,
                },
                "set_checkpoint": {
                    "directory": "results/SPIE 2021 1/.metadata/cga-ckpt-8",
                },
                "load": {
                    "filename": "results/SPIE 2021 1/.metadata/cgan-ckpt-8"
                },
            },
        }
