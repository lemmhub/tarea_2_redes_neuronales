import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
 #Leer datos
    a=np.array([[1,-1,-1,1]])
    b=np.array([[-1,1,-1,1]])
    c=np.array([[-1.,-1,1,-1]])
    arreglo=[a,b,c]
    hop =generar_Hopfield(arreglo)

    table=tabulate(hop,headers=[ i for i in  range(hop.shape[0])])
    print(table)

    recuperar(hop,arreglo)    

def generar_Hopfield(patrones):
    hopfield=np.zeros(patrones[0].shape[1])
    for patron in patrones:
        hopfield=hopfield+(np.dot(patron.T,patron)-np.eye(patron.shape[1]))
    return hopfield


def recuperar(Matriz,Patrones):
    for index,patron in enumerate(Patrones):
        recuperado=False
        iter=0
        while not recuperado:
            iter+=1
            patron_salida=np.dot(Matriz,patron.T)
            #Cambiamos valores de acuerdo a las condiciones
            patron_salida=np.where(patron_salida > 0,1,np.where(patron_salida<0,-1,0))
            if np.array_equal(patron,patron_salida.T):
                print(f"Se recuperó el patrón {index} en la iteración {iter}")
                recuperado=True
            patron=patron_salida.T

if __name__ == '__main__':
    main()