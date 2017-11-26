from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import Clasificador
import EstrategiaParticionado
from Datos import Datos
from PreprocesamientoAG import PreprocesamientoAG
from plotModel import plotModel
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    errores = []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=10,consApren=0.1)
    # error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador,45)
    #print error_media
    # p = PreprocesamientoAG(gener=(10,0.95))
    # c,f =p.seleccionarAtributos(dataset,clasificador)
    #
    # print c
    # print f



