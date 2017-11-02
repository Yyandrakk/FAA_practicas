from abc import ABCMeta, abstractmethod

import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances


class Clasificador(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # datosTrain: matriz numpy con los datos de entrenamiento
    # atributosDiscretos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
    # de variables discretas
    def entrenamiento(self, datosTrain, atributosDiscretos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada clasificador concreto
    # devuelve un numpy array con las predicciones
    def clasifica(self, datosTest, atributosDiscretos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    # TODO: implementar
    def error(self, datos, pred):
        return sum(map(lambda x, y: 0 if x == y else 1, datos[:, -1], pred)) / (len(datos[:, -1]) + 0.0)


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test
        particionado.creaParticiones(dataset.datos, seed)
        errores = np.array(())
        if len(particionado.particiones) == 1:
            clasificador.entrenamiento(dataset.extraeDatosTrain(particionado.particiones[0].indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
            dTrain = dataset.extraeDatosTest(particionado.particiones[0].indicesTest)
            clases = clasificador.clasifica(dTrain, dataset.nominalAtributos, dataset.diccionarios)
            return self.error(dTrain, clases), 0
        else:
            for particion in particionado.particiones:
                clasificador.entrenamiento(dataset.extraeDatosTrain(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionarios)
                dTrain = dataset.extraeDatosTest(particion.indicesTest)
                clases = clasificador.clasifica(dTrain, dataset.nominalAtributos, dataset.diccionarios)
                errores=np.append(errores,[self.error(dTrain, clases)])
            return errores.mean(), errores.std()



##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    def __init__(self, laplace=False):
        self.tablasV = []
        self.tablaC = {}
        self.laplace = laplace

    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        tam = len(diccionario)-1
        nClases = len(diccionario[-1])
        i = 0
        numFilas = datostrain.shape[0]
        if numFilas == 0:
            numFilas = 0.0000001
        for k in diccionario[-1].keys():
            v = diccionario[-1][k]
            self.tablaC[k] = datostrain[np.ix_(datostrain[:, -1] == v, (0, ))].shape[0] / (numFilas +0.0)

        while i < tam :

            if atributosDiscretos[i]:
                nAtri = len(diccionario[i])
                t = np.zeros((nAtri, nClases))
                for fila in datostrain:
                    t[int(fila[i]), int(fila[-1])] += 1
                if self.laplace and np.any(t==0):
                    t+=1

            else:
                t = np.zeros((2, nClases))
                for k in diccionario[-1].keys():
                    v = diccionario[-1][k]
                    t[0, int(v)] = np.mean(datostrain[np.ix_(datostrain[:, -1] == v, (i, ))])
                    t[1, int(v)] = np.var(datostrain[np.ix_(datostrain[:, -1] == v, (i ,))])

            self.tablasV.append(t)
            i += 1



    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):

        clases = []
        for fila in datostest:
           #i = 0
            posterior = {}
            for k in diccionario[-1].keys():
                v = diccionario[-1][k]
                aux = 1
                i = 0
                while i < (len(fila) - 1):
                    if atributosDiscretos[i]:
                        aux *= (self.tablasV[i][int(fila[i]), v] / sum(self.tablasV[i][:, v]))
                    else:
                        sqrt = math.sqrt(2*math.pi*self.tablasV[i][1,v])
                        exp = math.exp(-(((fila[i]-self.tablasV[i][0,v])**2)/(2.0*self.tablasV[i][1,v])))
                        aux *= (exp/sqrt)

                    i += 1
                aux = aux*self.tablaC[k]
                posterior[k] = aux

            clases.append(diccionario[-1][max(posterior,key=posterior.get)])

        return np.array(clases)


class ClasificadorVecinosProximos(Clasificador):

    def __init__(self, k=3):
        self.k = k
        self.listaMediasDesv = []
        self.datosTrainNormalizado = None

    def normalizarDatos(self, datos):
        i = 0
        tam = len(self.listaMediasDesv)-1
        aux = np.zeros(datos.shape)
        while i < tam:
            if self.listaMediasDesv[i]:
                aux[:,i] = (datos[:,i]-self.listaMediasDesv[i]["media"])/(self.listaMediasDesv[i]["desv"]+0.0)
            else:
                aux[:,i] = datos[:, i]
            i=i+1
        aux[:,i] = datos[:, i]

        return aux

    def calcularMediasDesv(self,datostrain,nCol):
            aux = {}
            aux["media"] = np.mean(datostrain[:,nCol])
            aux["desv"] = np.std(datostrain[:,nCol])
            self.listaMediasDesv.append(aux)

    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        tam = len(diccionario)-1
        i = 0

        while i < tam :

            if atributosDiscretos[i]:
                self.listaMediasDesv.append({})
            else:
                self.calcularMediasDesv(datostrain,i)
            i += 1

        self.datosTrainNormalizado = self.normalizarDatos(datostrain)

    def clasifica(self, datostest, atributosDiscretos, diccionario):

        datosNorm = self.normalizarDatos(datostest)
        tam = len(diccionario) - 1
        i = 0
        clases = []
        for fila in datosNorm:
            dstEu = euclidean_distances(self.datosTrainNormalizado, [fila]).tolist()

            aux = []

            for row in dstEu:
                aux.append(row[0])

            sortIndex = np.argsort(aux)

            KvecinosProximos=self.datosTrainNormalizado[sortIndex[0:self.k],-1]

            clases.append(np.bincount(KvecinosProximos.tolist()).argmax())

        return np.array(clases)
