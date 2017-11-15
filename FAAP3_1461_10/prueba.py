from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import Clasificador
import EstrategiaParticionado
from Datos import Datos
from plotModel import plotModel
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    errores =     []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    estrategia = EstrategiaParticionado.ValidacionCruzada()
    #clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=100,consApren=0.05)
   # error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador,45)
    #print error_media

    clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=100, consApren=0.01)
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador, 42)
    print error_media

    # encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
    # X = encAtributos.fit_transform(dataset.datos[:, :-1])
    # Y = dataset.datos[:, -1]
    #
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    #
    # # instantiate a logistic regression model, and fit with X and y
    # model = LogisticRegression(max_iter=10,class_weight=None)
    # pre=model.fit(x_train, y_train).predict(x_test)
    #
    # # check the accuracy on the training set
    # print 1 - accuracy_score(y_test, pre)
    # estrategia.creaParticiones(dataset.datos)
    #
    # ii = estrategia.particiones[-1].indicesTrain
    # clasificador.entrenamiento(dataset.extraeDatosTrain(ii),dataset.nominalAtributos,dataset.diccionarios)
    # plotModel(dataset.datos[ii, 0], dataset.datos[ii, 1], dataset.datos[ii, -1] != 0, clasificador, "Frontera", dataset.diccionarios)
    # #plt.figure()
    # plt.plot(dataset.datos[dataset.datos[:,-1]==0,0],dataset.datos[dataset.datos[:,-1]==0,1],'ro')
    # plt.plot(dataset.datos[dataset.datos[:,-1]==1,0],dataset.datos[dataset.datos[:,-1]==1,1],'bo')
    # plt.show()



