import random
import math
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

    def __binANum__(self,col):
        return np.flatnonzero(col)

    def __selProporcional__(self,pob,fitSum):
        sel = random.uniform(0, fitSum)
        acum = 0
        for c,f in pob:
            acum += f
            if acum > sel:
                return c


    def __seleccionProgenitores__(self,poblacion):
        sumFit=sum(cro[1] for cro in poblacion)
        return [self.__selProporcional__(poblacion,sumFit) for i in poblacion]


    def __fitPob__(self,colActive,dataset,clasificador):
        pobAux=[]
        estrategia = EstrategiaParticionado.ValidacionSimple()
        for col in colActive:
            dataSetAux = Datos()
            colNum = self.__binANum__(col)
            dataSetAux.datos = dataset.extraeDatosRelevantes(colNum)
            dataSetAux.diccionarios = dataset.diccionarioRelevante(colNum)
            dataSetAux.nominalAtributos = dataset.atribDiscretosRelevantes(colNum)
            e, _ = clasificador.validacion(estrategia, dataSetAux, clasificador)
            pobAux.append((col, 1 - e))
        pobAux.sort(key=lambda t: t[1], reverse=True)
        return pobAux

    def __cruceUniformePob__(self,pobAux):

        aux = []
        for p, s in zip(pobAux[0::2], pobAux[1::2]):
            pa,sa=self.__cruceUniforme__(p,s)
            aux.append(pa)
            aux.append(sa)

        return aux

    def __cruceUniforme__(self,p,s):
        for i in xrange(len(p)):
            if random.random() < self.pCruce:
                p[i], s[i] = s[i], p[i]
        return p,s

    def __mutacionPob__(self,pobAux):

        return [self.__mutacion__(c) for c in pobAux]

    def __mutacion__(self,c):
        for i in xrange(len(c)):
            if random.random() < self.pMut:
                c[i] = 0 if c[i]==1 else 1
        return c

    def __seleccionSup__(self,pobAux,poblacion):
        aux = []
        tamElite = math.ceil(self.pElitismo * self.tamPob)
        aux.extend(poblacion[0:tamElite])
        aux.extend(pobAux[0:(self.tamPob-tamElite)])
        return aux


    def seleccionarAtributos(self,dataset,clasificador):
        '''
        :param dataset:
        :type dataset: Datos
        :param clasificador:
        :type clasificador: Clasificador
        :return:
        '''


        generacionMax,pParada = self.geraciones
        columActive =np.unpackbits(np.random.randint(low=1,high=len(dataset.diccionarios), size=(self.tamPob,1),dtype=np.uint8),axis=1)
        poblacion = self.__fitPob__(columActive,dataset,clasificador)
        g=0
        p=0
        while g<generacionMax and all(lambda x: x[1]<pParada,poblacion):
            pobAux = self.__seleccionProgenitores__(poblacion)
            pobAux = self.__cruceUniformePob__(pobAux)
            pobAux = self.__mutacionPob__(pobAux)
            pobAux = self.__fitPob__(pobAux,dataset,clasificador)
            poblacion = self.__seleccionSup__(pobAux,poblacion)
            g+=1

        return self.__binANum__(poblacion[0][0])
