import Clasificador
import EstrategiaParticionado
from Datos import Datos

if __name__ == '__main__':
    dataset = Datos('./ConjuntosDatos/german.data')
    estrategia = EstrategiaParticionado.ValidacionCruzada(10)
    clasificador = Clasificador.ClasificadorNaiveBayes()
    errores = clasificador.validacion(estrategia, dataset, clasificador)
    print errores
