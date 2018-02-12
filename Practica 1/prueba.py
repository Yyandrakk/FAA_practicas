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
    dataset = Datos('./ConjuntosDatos/german.data')
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorNaiveBayes()
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)
    print error_media

    # Encode categorical integer features using a one-hot aka one-of-K scheme(categorical features)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]
    clf = GaussianNB()
    x_train, x_test, y_train,y_test = train_test_split(X,Y,test_size=0.4)
    predicciones=clf.fit(x_train,y_train).predict(x_test)
    print 1 - accuracy_score(y_test, predicciones)

    '''
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
    '''
