import Clasificador
import EstrategiaParticionado
from Datos import Datos
import numpy as np


class PreprocesamientoAG(object):

    def __init__(self,pCruce=0.6,pMut=0.001,pElitismo=0.05, tamPob=50, gener=(50,0.95)):
        self.pCruce = pCruce
        self.pMut=pMut
        self.pElitismo=pElitismo
        self.tamPob=tamPob
        self.geraciones=gener


    def seleccionarAtributos(self,dataset,clasificador):
        '''


        :param dataset:
        :type dataset: Datos
        :param clasificador:
        :type clasificador: Clasificador
        :return:
        '''
        if type(dataset) != Datos or type(clasificador) != Clasificador:
            raise TypeError, "dataset debe ser tipo Datos y clasificador de tipo Clasificador"

        generacionMax,pParada = self.geraciones
        poblacion =np.unpackbits(np.random.randint(low=1,high=len(dataset.diccionarios), size=(self.tamPob,1),dtype=np.uint8),axis=1)

        g=0
        while g<generacionMax and p<pParada:

            g+=1
