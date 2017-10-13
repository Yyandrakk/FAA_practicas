from abc import ABCMeta, abstractmethod
import numpy as np


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
        return sum(map(lambda x, y: 0 if x == y else 1, datos[:, -1], pred)) / len(datos[:, -1])


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
                errores.push(error(dTrain, clases))
                return np.mean(errores), np.std(errores)



##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    tablasV = []
    tablaC = {}

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
            self.tablaC[k] = datostrain[np.ix_(datostrain[:, -1] == v, (0, ))].shape[0] / numFilas

        while i < (tam - 1):

            if atributosDiscretos[i]:
                nAtri = len(diccionario[i])
                t = np.zeros((nAtri, nClases))
                for fila in datostrain:
                    t[fila[i], fila[-1]] += 1
            else:
                t = np.zeros((2, nClases))
                for k in diccionario[-1].keys():
                    v = diccionario[-1][k]
                    t[0, v] = np.mean(datostrain[np.ix_(datostrain[:, -1] == v, (i, ))])
                    t[1, v] = np.var(datostrain[np.ix_(datostrain[:, -1] == v, (i ,))])

            self.tablasV.append(t)
            i += 1

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):

        clases = []
        for fila in datostest:
            i = 0
            posterior = []
            for k in diccionario[-1].keys():
                v = diccionario[-1][k]
                aux = 1
                while i < (len(fila) - 1):
                    if atributosDiscretos:
                        aux=aux*(self.tablasV[i][int(fila[i]), v] / sum(self.tablasV[i][:, v]))
                    #else:
                    i+=1
                aux=aux*self.tablaC[k]
                posterior.append(aux)

            clases.append(max(posterior) in posterior)

        return np.array(clases)


        pass
