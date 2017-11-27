from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

import Clasificador
import EstrategiaParticionado
from Datos import Datos
from PreprocesamientoAG import PreprocesamientoAG
from plotModel import plotModel
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_regression
import numpy as np


if __name__ == '__main__':
    errores = []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=10,consApren=0.1)
    # error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador,45)
    #print error_media
    # p = PreprocesamientoAG(gener=(10,0.95))
    # c,f =p.seleccionarAtributos(dataset,clasificador)
    # print c
    # print f

    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:, :-1])
    Y = dataset.datos[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

    model = SelectKBest(chi2, k=10).fit(x_train, y_train)
    X_new = model.get_support() #se le puede pasar a fit X e Y
    print dataset.atribNombre([i for i, x in enumerate(X_new.tolist()) if x])

    lr = LogisticRegression(max_iter=10,class_weight=None).fit(x_train, y_train) #se le puede pasar a fit X e Y
    model = SelectFromModel(lr, prefit=True)
    #print model.estimator_.coef_
    X_new = model.get_support()
    print dataset.atribNombre([i for i, x in enumerate(X_new.tolist()) if x])

    model_filter = SelectKBest(f_regression, k=10)
    lr = LogisticRegression(max_iter=10,class_weight=None)
    model_pl = Pipeline([('SelectKBest', model_filter), ('LogisticRegression', lr)])
    model_pl.fit(X, Y)
    model_pl.predict(X)
    print model_pl.score(X, Y)
    print model_pl.named_steps['SelectKBest'].get_support()
    print dataset.atribNombre([i for i, x in enumerate(model_pl.named_steps['SelectKBest'].get_support().tolist()) if x])

    model_filter = SelectFromModel(lr, prefit=True)
    lr = LogisticRegression(max_iter=10,class_weight=None)
    model_pl = Pipeline([('SelectFromModel', model_filter), ('LogisticRegression', lr)])
    model_pl.fit(X, Y)
    model_pl.predict(X)
    print model_pl.score(X, Y)
    print model_pl.named_steps['SelectFromModel'].get_support()
    print dataset.atribNombre([i for i, x in enumerate(model_pl.named_steps['SelectKBest'].get_support().tolist()) if x])


