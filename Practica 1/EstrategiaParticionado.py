import math
import random
from abc import ABCMeta, abstractmethod


class Particion():

    indicesTrain = []
    indicesTest = []

    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []

#####################################################################################################


class EstrategiaParticionado(object):

    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta
    nombreEstrategia = "null"
    numeroParticiones = 0
    particiones = []

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos, seed=None):
        pass


#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

    def __init__(self):
        self.nombreEstrategia = "ValidacionSimple"

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos, seed=None):
        random.seed(seed)
        numFilas = datos.shape[0]
        particion = Particion()
        index = list(xrange(0, numFilas-1))
        random.shuffle(index)
        numTrain = int(math.ceil(numFilas * self.numeroParticiones))
        particion.indicesTrain = index[0: numTrain]
        particion.indicesTest = index[numTrain + 1:]
        self.particiones.append(particion)

#####################################################################################################


class ValidacionCruzada(EstrategiaParticionado):

    def __init__(self, k):
        self.nombreEstrategia = "ValidacionCruzada"
        self.numeroParticiones = k
    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones
    # y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar

    def creaParticiones(self, datos, seed=None):
        random.seed(seed)
        numFilas = datos.shape[0]
        index = list(xrange(0, numFilas))
        random.shuffle(index)
        indexK = numFilas // self.numeroParticiones
        lIndex = [index[i:i + indexK] for i in xrange(0, len(index), indexK)]
        i = 0
        while i < self.numeroParticiones:
            particion = Particion()
            particion.indicesTest = lIndex[i]
            auxL = [x for j, x in enumerate(lIndex) if j != i]
            particion.indicesTrain = [x for subL in auxL for x in subL]
            self.particiones.append(particion)
            i += 1
