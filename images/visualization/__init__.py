import cv2
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt

def show(images: np.ndarray, titles:List[List[str]] = None, rows=10, columns = 5, figsize: Dict[int, int] = (12,8)):
    def index_in_list(lst, index):
        return index < len(list)
    
    if len(images.shape) == 3 and images.shape[-1] == 1:
        plt.axis('off')
        plt.figure(figsize=figsize)
        plt.imshow(images[:,:,0], cmap='gray', interpolation='nearest')

        if type(titles) == str:
            plt.title(titles)
        return
    elif len(images.shape) == 4:
        if titles is not None and type(titles) == list:
            set_title = True

        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        
        for i in range(rows):
            for j in range(columns):
                ax[i,j].axis('off')

                if set_title and index_in_list(title, i) and index_in_list(title[i], j):
                    ax[i,j].set_title(titulos[i])
                
                ax[i,j].imshow(images[i*max_rows + j, :,:, 0], cmap='gray', interpolation='nearest')

def show_images(*args):
    fig, ax = plt.subplots(1, len(args), figsize=(12,8))

    for i, img in enumerate(args):
        ax[i].axis('off')
        ax[i].imshow(img[:, :, 0], cmap='gray', interpolation='nearest')


def show_single_image(img):
    assert len(img.shape) == 2, 'A imagem deve ter apenas 2 dimensões.'

    plt.figure(figsize=(12,8))
    plt.imshow(img, cmap='gray', interpolation='nearest')

def mostrar_lado_a_lado(imagens: List[np.ndarray], titulos: List[str], figsize: Dict[int, int] = (12,8)):
    """
    Imprime as imagens que estiverem na lista com os títulos apresentados.
    """
    
    assert len(imagens) == len(titulos), 'imagens e titulos devem ter o mesmo tamanho.'
    assert len(imagens[0].shape) == 2, 'As imagens deve ter apenas 2 dimensões.'
    
    quantidade = len(imagens)
    
    fig, ax = plt.subplots(1, quantidade, figsize=figsize)
    
    for i in range(quantidade):
        ax[i].axis('off')
        ax[i].set_title(titulos[i])
        ax[i].imshow(imagens[i], cmap='gray', interpolation='nearest')
