import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

def main():
    #Leer datos
    datos = pd.read_csv("1_adaline.csv")
    datos_py = np.array(datos)
    x=datos_py[:,1:4]
    y=datos_py[:,4]
    y=y.reshape(y.shape[0],1)

    #Parámetros e hiperparámetros
    np.random.seed(20)
    w=np.random.randn(x.shape[1],1) #variables,neuronas

    alfa=0.02
    epochs=0
    lim_epochs=100000
    precision=0.0000001
    error_total=[precision+1]
    error_muestra=np.empty([y.shape[0],1])

    while (precision<error_total[-1] and epochs < lim_epochs):
        #Realizamos la actualización de los pesos por muestra (11 en este ejemplo)
        for i in range(y.shape[0]):
            y_hat = np.dot(x[i,:].reshape(1,x.shape[1]),w)
            #obtener delta
            delta = y_hat - y[i]
            #Gradiente por pesos
            grad=delta*x[i,:]

            #modificar los pesos incluyendo alfa
            grad=-grad*alfa
            w=(w+grad.T)

            error_muestra[i,0]=(delta**2)/2
        #se calcula el Error cuadrático medio
        error_total.append(np.mean(error_muestra))
        epochs+=1
        
    print(f'Error final {error_total[-1]}')
    plt.plot(error_total[1:28],marker='o')
    plt.xlabel("Época")
    plt.ylabel("Costo")
    plt.show()
   
    print(f'Pesos finales {w.T}')
    print(f'Épocas {epochs}')
    return w
    
def probar_pesos(w):
    #Leer datos
    datos = pd.read_csv("1_b.csv")
    datos_py = np.array(datos)
    x=datos_py[:,1:4]
    y=datos_py[:,4]
    y=y.reshape(y.shape[0],1)
    y_hat = np.dot(x,w)

    valores_tabulate=[]
    for i in range(len(y_hat)):
        valores_tabulate.append([y_hat[i,0],y[i,0],np.abs(y_hat[i,0]-y[i,0])])

    print(tabulate(valores_tabulate,headers=["Predicción", "Etiqueta", "Error Absoluto"]))



if __name__ == '__main__':
    w=main()
    probar_pesos(w)