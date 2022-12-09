import numpy as np
import pandas as pd
from tabulate import tabulate
import pandas as pd

def main():
    tabla_recup=[ ]
 #Leer datos
    matriz_cuad_size=35
    datos = pd.read_csv("3_patrones.csv")
    datos_py = np.array(datos)
    #modificar 0 por -1
    datos_py[datos_py==0]=-1
    datos_py = datos_py.ravel('F')
    datos_py=datos_py[~np.isnan(datos_py)]      

    #Separando datos
    X=[]
    for i in range(5):
        X.append(np.array([datos_py.ravel('F')[matriz_cuad_size*i:matriz_cuad_size*(i+1)]]))
    #Se trajo como columnas, entonces buscamos ese orden (5,7)
    #para imprimir se regresa la transpuesta
    #print(X[1].reshape(5,7).T)
    
    #Generando la matriz de Hopfield
    M_Hopfield=generar_Hopfield(X)
    table=tabulate(M_Hopfield,headers=[ i for i in  range(matriz_cuad_size)])
    print(table)

    #Fase de recuperación
    #Recuperando original
    #tabla_recup.append(recuperar(M_Hopfield,X,X))
    #Convergencia con el anterior
    tabla_recup.append(recuperar(M_Hopfield,X))

    #Modificar patrones
    X_old=np.copy(X)
    for i in range(1,6):
        X_new,X_old=agregar_ruido(X_old,i)
        print(f"Modificando {i} pixeles: ************************")
    #Recuperando original
        #tabla_recup.append(recuperar(M_Hopfield,X_new,X_old))
        #Convergencia con el anterior
        tabla_recup.append(recuperar(M_Hopfield,X_new))

    
    df=pd.DataFrame( [index]+i for index,i in enumerate(tabla_recup) )
    df.columns=["Pixeles con ruido","Iter. Patrón 1","Iter. Patrón 2","Iter. Patrón 3","Iter. Patrón 4","Iter. Patrón 5"]
    print(df)

def generar_Hopfield(patrones):
    hopfield=np.zeros(patrones[0].shape[1])
    for patron in patrones:
        hopfield=hopfield+(np.dot(patron.T,patron)-np.eye(patron.shape[1]))
    return hopfield



def recuperar(Matriz,Patrones,patron_comparar=0):
    iter_max_recup=100000
    iteraciones_recup=[]
    for index,patron in enumerate(Patrones):
        recuperado=False
        iter=0
        while not recuperado and iter<iter_max_recup:
            iter+=1
            patron_salida=np.dot(Matriz,patron.T)
            #Cambiamos valores de acuerdo a las condiciones
            patron_salida=np.where(patron_salida > 0,1,np.where(patron_salida<0,-1,0))
            #Revisamos si comparamos con el original o con el anterior inmediato
            if patron_comparar==0:
                if np.array_equal(patron,patron_salida.T):
                    print(f"Se recuperó el patrón {index} en la iteración {iter}")
                    recuperado=True
                    iteraciones_recup.append(iter)
            else:
                if np.array_equal(patron_comparar[index],patron_salida.T):
                    print(f"Se recuperó el patrón {index} en la iteración {iter}")
                    recuperado=True
                    iteraciones_recup.append(iter)

            patron=patron_salida.T
            if iter==iter_max_recup and not recuperado:
                print(f"No se recuperó el patrón {index}")
                iteraciones_recup.append("No recup")

    return iteraciones_recup

def agregar_ruido(patrones,pixeles):
    patrones_old=np.copy(patrones)
    #Generador
    rng=np.random.default_rng()
    for patron in patrones:
        if patron.shape[1]<pixeles:
            print("El número de pixeles excede el tamaño del patrón")
            return
        indices=rng.choice(patron.shape[1],size=pixeles, replace=False)
        patron[0][indices]=patron[0][indices]*-1
        
    return patrones,patrones_old

if __name__ == '__main__':
    main()