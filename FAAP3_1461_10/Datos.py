import numpy as np


class Datos(object):

    TiposDeAtributos = ('Continuo', 'Nominal')

    # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos,nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero=None):
        self.tipoAtributos = []
        self.nombreAtributos = []
        self.nominalAtributos = []
        self.datos = np.array(())
        self.diccionarios = []
        if nombreFichero == None:
            return
        with open(nombreFichero, 'r') as f:
            linesL = f.read().splitlines()
            con, nom = self.TiposDeAtributos
            self.nDatos = int(linesL[0])
            self.nombreAtributos = linesL[1].split(',')
            self.tipoAtributos = linesL[2].split(',')
            self.nominalAtributos = map(lambda x: True if x == nom else False, self.tipoAtributos)
            datosAux = np.array(map(lambda x: list(x.split(',')), linesL[3:]))
            i = 0
            tam = len(self.nominalAtributos)
            while i < tam:
                if not self.nominalAtributos[i]:
                    self.diccionarios.append({})
                else:
                    auxC = set(datosAux[:, i])
                    j = 0
                    dicAux = {}
                    for x in sorted(auxC):
                        dicAux[x] = j
                        j += 1
                    self.diccionarios.append(dicAux)
                i += 1

            i = 0
            self.datos = np.empty((self.nDatos, tam))
            while i < tam:
                if not self.nominalAtributos[i]:
                    self.datos[:, i] = datosAux[:, i]
                else:
                    dic = self.diccionarios[i]
                    j = 0
                    while j < self.nDatos:
                        self.datos[j, i] = dic[datosAux[j, i]]
                        j += 1
                i += 1

# TODO
    def extraeDatosTrain(self, idx):
        return self.datos[idx, :]

    def extraeDatosTest(self, idx):
        return self.datos[idx, :]

    def extraeDatosRelevantes(self,idx):
        return self.datos[:,np.append(idx,len(self.diccionarios)-1)]

    def diccionarioRelevante(self,idx):
        aux = [ self.diccionarios[i] for i in idx]
        aux.append(self.diccionarios[-1])
        return aux

    def atribDiscretosRelevantes(self,idx):
        aux= [ self.nominalAtributos[i] for i in idx]
        aux.append(self.nominalAtributos[-1])
        return aux

    def atribNombre(self, idx):
        return [self.nombreAtributos[i] for i in idx]