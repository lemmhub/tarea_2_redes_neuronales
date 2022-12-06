import numpy as np
import pandas as pd

def main():

 #Leer datos
    datos = pd.read_csv("3_patrones.csv")
    datos_py = np.array(datos)
    #modificar 0 por -1
    datos_py[datos_py==0]=-1
    datos_py = datos_py.ravel('F')
    datos_py=datos_py[~np.isnan(datos_py)]         

    #Separando datos
    X=[]
    for i in range(5):
        X.append(np.array([datos_py.ravel('F')[35*i:35*(i+1)]]))
    #Se trajo como columnas, entonces buscamos ese orden (5,7)
    #para imprimir se regresa la transpuesta
    #print(X[1].reshape(5,7).T)
    
    #Generando la matriz de Hopfield
    M_Hopfield=generar_Hopfield(X)
    #print(M_Hopfield)      

    #prueba Sossa
    a=np.array([[1,-1,-1,1]])
    b=np.array([[-1,1,-1,1]])
    c=np.array([[-1,-1,1,-1]])
    arreglo=[a,b,c]
    hop =generar_Hopfield(arreglo)
    print(hop)
    #sí jala
    

def generar_Hopfield(patrones):
    hopfield=np.zeros(patrones[0].shape[1])
    for patron in patrones:
        hopfield=hopfield+(np.dot(patron.T,patron)-np.eye(patron.shape[1]))
    return hopfield






if __name__ == '__main__':
    main()
