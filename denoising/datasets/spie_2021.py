import os
import numpy as np
from typing import List, Dict, Tuple

def adiciona_a_dimensao_das_cores(array:np.ndarray) -> np.ndarray:
    """
    Adiciona a dimensão das cores no array numpy, considerando a imagem sendo escala de cinza.
    """
    return array.reshape( array.shape + (1,) )

def dividir_dataset_em_treinamento_e_teste(dataset: np.ndarray, divisao=(80,20)):
    """
    Divisão representa a porcentagem entre conj. de treinamento e conj. de teste.
    Ex: (80,20) representa 80% para treino e 20% para teste.
    """
    assert len(divisao) == 2, 'Divisão deve ser: % de conj. de treinamento e % de conj. de teste.'
    
    n_treino, n_teste = divisao
    
    assert n_treino + n_teste == 100, 'A soma da divisão deve ser igual a 100.'
    
    total = dataset.shape[0] 
    porcentagem_treino = n_treino/100 #0.8
    porcentagem_teste = n_teste/100 #0.2
    
    return dataset[:int(porcentagem_treino*total)], dataset[int(porcentagem_treino*total):]

def carrega_dataset(diretorio_datasets: str, divisao: Tuple[int, int], embaralhar=True):    
    x = np.load(os.path.join(diretorio_datasets, 'noisy.npy'))
    y = np.load(os.path.join(diretorio_datasets, 'original.npy'))

    if embaralhar:
        np.random.seed(42)
        np.random.shuffle(x)
        np.random.seed(42)
        np.random.shuffle(y)

    x_train, x_test = dividir_dataset_em_treinamento_e_teste(x, divisao=divisao)
    y_train, y_test = dividir_dataset_em_treinamento_e_teste(y, divisao=divisao)

    return (x_train, y_train, x_test, y_test)