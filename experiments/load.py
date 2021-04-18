import yaml
from typing import List
from .decorator import method

from methods import Methods

class Method:
    def __init__(self, name:str, instance, need_train:bool = False, *args, **kwargs):
        self.name = name
        self.instance = instance
        self.need_train = need_train

        self.__dict__.update(kwargs)

        self.images = None
        self.psnr = None
        self.ssim = None
        self.runtime = None

    @property
    def is_traditional(self):
        return (self.need_train == False)

    def __str__(self):
        return self.name

def load_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        return config

def load_methods() -> List[Method]:
    """Read all methods from methods.py with the @method decorator.
    Convert the dict response in a Method object 
    and return the methods list. 

    Returns:
        list[Method]: return the methods list to be used in the experiment.
    """
    functions = method.methods(Methods)
    new_methods = []

    for k, func in functions.items():
        dict_method = func()

        assert isinstance(dict_method, dict), f"return of {k}() should be a dict."
        assert 'name' in dict_method, f"name attribute is required in {k}() return."
        assert 'instance' in dict_method, f"instance attribute is required in {k}() return."

        new_method = Method(**dict_method)
        new_methods.append(new_method)
    
    return new_methods
