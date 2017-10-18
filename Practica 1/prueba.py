import Clasificador
import EstrategiaParticionado
from Datos import Datos
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np

if __name__ == '__main__':
    errores =     []
    dataset = Datos('./ConjuntosDatos/german.data')
    estrategia = EstrategiaParticionado.ValidacionCruzada()
    clasificador = Clasificador.ClasificadorNaiveBayes()
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)
    print error_media

    # Encode categorical integer features using a one-hot aka one-of-K scheme(categorical features)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]
    clf = GaussianNB()
    score = cross_val_score(clf, X, Y, cv=10)
    print 1-score.mean(), score.std()

    dataset2 = Datos('./ConjuntosDatos/tic-tac-toe.data')
    estrategia2 = EstrategiaParticionado.ValidacionCruzada()
    clasificador2 = Clasificador.ClasificadorNaiveBayes()
    error_media, error_std = clasificador.validacion(estrategia2, dataset2, clasificador2)
    print error_media

    # Encode categorical integer features using a one-hot aka one-of-K scheme(categorical features)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset2.nominalAtributos[:-1], sparse=False)
    X2 = encAtributos.fit_transform(dataset2.datos[:, :-1])
    Y2 = dataset2.datos[:, -1]
    clf2 = MultinomialNB()
    score = cross_val_score(clf2, X2, Y2, cv=10)
    print 1 - score.mean(), score.std()
