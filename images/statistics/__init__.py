import cv2
from typing import List
import numpy as np
import matplotlib.pyplot as plt

def colored_histogram(imagem, intervalo=(0, 256)):
    """ 
    Histograma para imagem colorida (RGB).
    """
    
    color = ('b','g','r')
    
    fig, ax = plt.subplots(3,1, figsize=(12,8))
    
    for i,col in enumerate(color):
        histr = cv2.calcHist([imagem],[i],None,[intervalo[1]],[intervalo[0],intervalo[1]])
        ax[i].plot(histr, color = col)
        ax[i].set_xlim([intervalo[0],intervalo[1]])
#         plt.plot(histr,color = col)
#         plt.xlim([intervalo[0],intervalo[1]])
    plt.show()

def histogram(imagem, intervalo=(0, 256)):
    plt.figure(figsize=(12,8))
    
    histr = cv2.calcHist([imagem],[0],None,[intervalo[1]],[intervalo[0],intervalo[1]])
    plt.plot(histr, color = 'b')
    plt.xlim([intervalo[0],intervalo[1]])
    plt.show()