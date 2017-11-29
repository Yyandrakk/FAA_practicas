from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import Clasificador
from Datos import Datos
from PreprocesamientoAG import PreprocesamientoAG
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, f_regression, f_classif


if __name__ == '__main__':
    errores = []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    clasificador = Clasificador.ClasificadorVecinosProximos()
    # error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador,45)
    #print error_media
    p = PreprocesamientoAG( gener=(50,0.995))
    c,f =p.seleccionarAtributos(dataset,clasificador)
    print c
    print f


    print '\nSelectKBest'
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1], sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:, :-1])
    Y = dataset.datos[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

    # model = SelectKBest(chi2, k=10).fit(x_train, y_train)
    # X_new = model.get_support() #se le puede pasar a fit X e Y
    # print dataset.atribNombre([i for i, x in enumerate(X_new.tolist()) if x])

#     lr = LogisticRegression(max_iter=10,class_weight=None).fit(x_train, y_train) #se le puede pasar a fit X e Y
#     model = SelectFromModel(lr, prefit=True)
# #    print model.estimator_.coef_
#     X_new = model.get_support()
#     print dataset.atribNombre([i for i, x in enumerate(X_new.tolist()) if x])
#     #
    model_filter = SelectKBest(f_classif, k=10)
    lr = LogisticRegression(max_iter=100,class_weight=None)
    model_pl = Pipeline([('SelectKBest', model_filter), ('LogisticRegression', lr)])
    model_pl.fit(x_train, y_train)
    model_pl.predict(x_test)
    print model_pl.score(x_test, y_test)
    print model_pl.named_steps['SelectKBest'].get_support()
    print dataset.atribNombre([i for i, x in enumerate(model_pl.named_steps['SelectKBest'].get_support().tolist()) if x])

    lr = LogisticRegression(max_iter=100,class_weight=None)
    lrS = LogisticRegression(max_iter=100, class_weight=None).fit(x_train, y_train)
    model_filter = SelectFromModel(lrS)

    print '\nSelectFromModel'
    model_pl = Pipeline([('SelectFromModel', model_filter), ('LogisticRegression', lr)])
    model_pl.fit(x_train, y_train)
    model_pl.predict(x_test)
    print model_pl.score(x_test, y_test)
    print model_pl.named_steps['SelectFromModel'].get_support()
    print dataset.atribNombre([i for i, x in enumerate(model_pl.named_steps['SelectFromModel'].get_support().tolist()) if x])

