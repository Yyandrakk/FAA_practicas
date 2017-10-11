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
        return sum(map(lambda x, y: 0 if x == y else 1, datos, pred)) / len(datos)


    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    # TODO: implementar esta funcion
    def validacion(self, particionado, dataset, clasificador, seed=None):

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
        # y obtenemos el error en la particion de test i
        # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
        # y obtenemos el error en la particion test
        particionado.creaParticiones(dataset, seed)
        errores = np.array(())
        if len(particionado.particiones) == 1:
            clasificador.entrenamiento(dataset.extraeDatosTrain(particionado.particiones[0].indicesTrain), dataset.nominalAtributos, dataset.diccionario)
            dTrain = dataset.extraeDatosTest(particionado.particiones[0].indicesTest)
            clases = clasificador.clasifica(dTrain, dataset.nominalAtributos, dataset.diccionario)
            return error(dTrain, clases), 0
        else:
            for particion in particionado.particiones:
                clasificador.entrenamiento(dataset.extraeDatosTrain(particion.indicesTrain), dataset.nominalAtributos, dataset.diccionario)
                dTrain = dataset.extraeDatosTest(particion.indicesTest)
                clases = clasificador.clasifica(dTrain, dataset.nominalAtributos, dataset.diccionario)
                errores.push(error(dTrain, clases))
                return np.mean(errores), np.std(errores)



##############################################################################

class ClasificadorNaiveBayes(Clasificador):

    tablas = []
    # TODO: implementar
    def entrenamiento(self, datostrain, atributosDiscretos, diccionario):

        tam = len(diccionario)
        nClases = len(diccionario[-1])
        i=0
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

            self.tablas.push(t)
            i += 1

    # TODO: implementar
    def clasifica(self, datostest, atributosDiscretos, diccionario):
        pass
