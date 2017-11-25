import random
import math
import Clasificador
import EstrategiaParticionado
from Datos import Datos
import numpy as np
from itertools import chain


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
        return [self.__selProporcional__(poblacion,sumFit) for _ in poblacion]

    def __fit__(self,col,dataset,clasificador,estrategia):
        dataSetAux = Datos()
        colNum = self.__binANum__(col)
        dataSetAux.datos = dataset.extraeDatosRelevantes(colNum)
        dataSetAux.diccionarios = dataset.diccionarioRelevante(colNum)
        dataSetAux.nominalAtributos = dataset.atribDiscretosRelevantes(colNum)
        e, _ = clasificador.validacion(estrategia, dataSetAux, clasificador)
        return col, 1 - e

    def __fitPob__(self,colActive,dataset,clasificador):
        estrategia = EstrategiaParticionado.ValidacionSimple()
        return sorted([self.__fit__(col,dataset,clasificador,estrategia) for col in colActive],key=lambda t: t[1], reverse=True)

    def __cruceUniformePob__(self,pobAux):
        return list(chain.from_iterable((self.__cruceUniforme__(p,s) for p, s in zip(pobAux[0::2], pobAux[1::2]))))

    def __cruceUniforme__(self,p,s):
        for i in xrange(len(p)):
            if random.random() < self.pCruce:
                p[i], s[i] = s[i], p[i]

        if all(x==0 for x in p):
            p[random.ranint(0,len(p)-1)]=1
        if all(x==0 for x in s):
            s[random.ranint(0,len(s)-1)]=1
        return p,s

    def __mutacionPob__(self,pobAux):
        return [self.__mutacion__(c) for c in pobAux]

    def __mutacion__(self,c):
        for i in xrange(len(c)):
            if random.random() < self.pMut:
                c[i] = 0 if c[i]==1 else 1

        if all(x==0 for x in c):
            c[random.ranint(0,len(c)-1)]=1
        return c

    def __seleccionSup__(self,pobAux,poblacion):
        aux = []
        tamElite = int(math.ceil(self.pElitismo * self.tamPob))
        aux.extend(poblacion[0:tamElite])
        aux.extend(pobAux[0:(self.tamPob-tamElite)])
        return sorted(aux,key=lambda t: t[1], reverse=True)


    def seleccionarAtributos(self,dataset,clasificador):
        '''
        :param dataset:
        :type dataset: Datos
        :param clasificador:
        :type clasificador: Clasificador
        :return:
        '''
        generacionMax,pParada = self.geraciones
        #columActive =np.unpackbits(,axis=1)
        tam = len(dataset.diccionarios)
        intAle = np.random.randint(low=1,high=tam, size=(self.tamPob,1),dtype=np.uint64)
        binarios = np.array([np.hstack((np.ones(n, dtype=np.uint64), np.zeros(tam - n - 1, dtype=np.uint64))) for n in intAle])
        binariosAle = np.array(binarios)
        map(lambda x: np.random.shuffle(x),binariosAle)
        poblacion = self.__fitPob__(binariosAle,dataset,clasificador)
        g=0
        while g<generacionMax and all(p[1]<pParada for p in poblacion):
            pobAux = self.__seleccionProgenitores__(poblacion)
            pobAux = self.__cruceUniformePob__(pobAux)
            pobAux = self.__mutacionPob__(pobAux)
            pobAux = self.__fitPob__(pobAux,dataset,clasificador)
            poblacion = self.__seleccionSup__(pobAux,poblacion)
            print "####################################"
            print g
            g+=1

        return self.__binANum__(poblacion[0][0]), poblacion[0][1]
