import Clasificador
import EstrategiaParticionado
from Datos import Datos
from plotModel import plotModel
import matplotlib.pyplot as plt

if __name__ == '__main__':
    errores =     []
    dataset = Datos('./ConjuntoDatos/example1.data')
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=10)
    error_media, error_std = clasificador.validacion(estrategia, dataset, clasificador)
    print error_media
    ii = estrategia.particiones[-1].indicesTrain
    plotModel(dataset.datos[ii, 0], dataset.datos[ii, 1], dataset.datos[ii, -1] != 0, clasificador, "Frontera", dataset.diccionarios)
    plt.figure()
    plt.plot(dataset.datos[dataset.datos[:,-1]==0,0],dataset.datos[dataset.datos[:,1]==0,1],'bo')
    plt.plot(dataset.datos[dataset.datos[:,-1]==1,0],dataset.datos[dataset.datos[:,1]==1,1],'ro')

