import numpy as np
import sys

class Datos(object):

    TiposDeAtributos = ('Continuo', 'Nominal')
    tipoAtributos = []
    nombreAtributos = []
    nominalAtributos = []
    datos = np.array(())
    # Lista de diccionarios. Uno por cada atributo.
    diccionarios = []
    # TODO: procesar el fichero para asignar correctamente las variables
    #  tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios

    def __init__(self, nombreFichero):
            print self.diccionarios
            print self.datos
            print type(self.datos[0,1])
    # TODO: implementar en la prctica 1

    def extraeDatosTrain(idx):
        pass

    def extraeDatosTest(idx):
        pass

if __name__ == '__main__':
    datos= Datos(sys.argv[1])
