from sklearn.neighbors import KNeighborsClassifier

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
    errores = []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    estrategia = EstrategiaParticionado.ValidacionCruzada()
    clasificador = Clasificador.ClasificadorVecinosProximos()
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)
    print error_media

    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:, :-1])
    Y = dataset.datos[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='euclidean')

    knn.fit(x_train, y_train)

    pred = knn.fit(x_train, y_train).predict(x_test)

    print 1 - accuracy_score(y_test, pred)