import Clasificador
import EstrategiaParticionado
from Datos import Datos
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    dataset = Datos('./ConjuntosDatos/german.data')
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorNaiveBayes()
    errores = clasificador.validacion(estrategia, dataset, clasificador)
    print errores


    # Encode categorical integer features using a one-hot aka one-of-K scheme(categorical features)
    encAtributos = preprocessing.OneHotEncoder(categorical_features=dataset.nominalAtributos[:-1],sparse=False)
    X = encAtributos.fit_transform(dataset.datos[:,:-1])
    Y = dataset.datos[:,-1]
    clf = GaussianNB()
    clf.fit(X, Y)
    GaussianNB(priors=None)
    print cross_val_score(clf, X, Y)