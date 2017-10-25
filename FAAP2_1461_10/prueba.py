import Clasificador
import EstrategiaParticionado
from Datos import Datos
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

if __name__ == '__main__':
    errores =     []
    dataset = Datos('./ConjuntoDatos/example1.data')
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorVecinosProximos()
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)
    print error_media

