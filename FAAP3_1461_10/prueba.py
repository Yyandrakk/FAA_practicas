import Clasificador
import EstrategiaParticionado
from Datos import Datos
from PreprocesamientoAG import PreprocesamientoAG

if __name__ == '__main__':
    errores = []
    dataset = Datos('./ConjuntoDatos/wdbc.data')
    estrategia = EstrategiaParticionado.ValidacionSimple()
    clasificador = Clasificador.ClasificadorRegresionLogistica(nEpoc=10, consApren=1)

    prepro = PreprocesamientoAG()
    prepro.seleccionarAtributos(dataset, clasificador)




