import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

def main():
        #Leer datos
    datos = pd.read_csv("2_and_or.csv")
    datos_py = np.array(datos)
    #Caso and
    x=datos_py[:4,:2]    
    y_and=datos_py[:4,2]
    y_and=y_and.reshape(y_and.shape[0],1)

    #Caso or
    y_or=datos_py[4:,2]
    y_or=y_or.reshape(y_or.shape[0],1)

    #Aumentando x
    x = np.array([[a,b,-1] for a,b in x])

    #Parámetros e hiperparámetros
    np.random.seed(12)
    w_and=np.array([[0.1,0.2,0.3]]) #variables,neuronas
    w_or=np.array([[0.1,0.2,0.3]]) #variables,neuronas
    
    alfa=0.02
    epochs=0
    lim_epochs=10000
    precision=0.00000001
    error_total_and=[precision+1]
    error_total_or=[precision+1]
    
    w_and_gif=[]
    w_or_gif=[]
    #Se crearan 10 imgs para el gif y un compendio inicial continuo en donde se aprecian más cambios
    intervalo_gif=lim_epochs/10
    limite_inicial_gif=100

    #se puede utilizar cualquier "y"
    error_muestra_and=np.empty([y_and.shape[0],1])
    error_muestra_or=np.empty([y_and.shape[0],1])

    encontrado_and=False
    encontrado_or = False
    epoca_encontrado_and=0
    epoca_encontrado_or=0

    #datos de impresión 1.w inicial, 2.a la primera solución, 3.10 épocas, 4.al límite de épocas/error
    w_print = np.empty([2,4],dtype="object")
    w_print[0,0]=w_and
    w_print[1,0]=w_or
    
    while ( epochs < lim_epochs):
        #Realizamos la actualización de los pesos por muestra
        for i in range(y_and.shape[0]):
            a_and = np.dot(x[i,:],w_and.T)
            a_or = np.dot(x[i,:],w_or.T)

            y_hat_and = np.tanh(a_and)
            y_hat_or = np.tanh(a_or)       
            
            #obtener delta
            delta_and = y_hat_and - y_and[i]
            delta_or = y_hat_or - y_or[i]            

            #Gradiente por pesos
            grad_and= delta_and*(1-(y_hat_and**2))*x[i,:]
            grad_or= delta_or*(1-(y_hat_or**2))*x[i,:]
            #modificar los pesos incluyendo alfa
            #print(f'grad_and {grad_and}')

            grad_and = -grad_and*alfa
            grad_or = - grad_or*alfa

            w_and=(w_and+grad_and)
            w_or=(w_or+grad_or)

            error_muestra_and[i,0]=(delta_and**2)/2
            error_muestra_or[i,0]=(delta_or**2)/2

        #se calcula el Error cuadrático medio
        error_total_and.append(np.mean(error_muestra_and))
        error_total_or.append(np.mean(error_muestra_or))
        
        if not encontrado_and:
            encontrado_and=probar_sol(w_and,x,y_and)
            if encontrado_and:
                print(f'Se alcanzó la solución del caso AND en la época {epochs} con pesos {w_and}')
                epoca_encontrado_and=epochs
                w_print[0,1]=w_and

        if not encontrado_or:
            encontrado_or=probar_sol(w_or,x,y_or)
            if encontrado_or:
                print(f'Se alcanzó la solución del caso OR en la época {epochs} con pesos {w_or}')
                epoca_encontrado_or=epochs
                w_print[1,1]=w_or
            
        if epochs==10:
            w_print[0,2]=w_and
            w_print[1,2]=w_or

        if epochs%intervalo_gif==0 or epochs<limite_inicial_gif:
            w_and_gif.append(w_and)
            w_or_gif.append(w_or)
        
        epochs+=1

    print(f'Épocas {epochs}')
    print("CASO AND")

    print(f'Error final caso AND {error_total_and[-1]}')
    print(f'Pesos finales AND {w_and}')
    print("")
    print("CASO OR")

    print(f'Error final caso OR {error_total_or[-1]}')
    print(f'Pesos finales OR {w_or}')

    plt.plot(error_total_and[1:28],marker='o')
    plt.title("AND")
    plt.xlabel("Época")
    plt.ylabel("Costo")
    plt.show()

    plt.plot(error_total_or[1:28],marker='8',color="red")
    plt.title("OR")
    plt.xlabel("Época")
    plt.ylabel("Costo")
    plt.show()
    
    #pesos al final de la corrida error/lim epochs
    w_print[0,3]=w_and
    w_print[1,3]=w_or
    epochs_encontrada=[epoca_encontrado_and,epoca_encontrado_or]

    graficar_pesos(w_print,x,epochs_encontrada,epochs)    
    
    #crear_gif(w_and_gif,x,"and")
    #crear_gif(w_or_gif,x,"or")

    return

def probar_sol(w,x,y):
    y_hat=np.tanh( np.dot(x,w.T))
    #Esperamos que todos sean positivos
    tabla_verdad =(y*y_hat)>0
    return tabla_verdad.all()

def graficar_pesos(w,x,epochs_encontrada,epochs_totales):
    for index,i in enumerate(w):
        plt.scatter(x[:,0],x[:,1],marker="o")
        x_print=np.linspace(-0.5,1.5,3)
        if index==0:
            plt.scatter(1,1,marker="o",color="red")
            plt.title('Caso AND')

        if index==1:
            plt.scatter(0,0,marker="o",color="red")
            plt.title('Caso OR')
        
        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.xlim(-0.5,1.5)
        plt.ylim(-0.5,1.5)

        plt.plot(x_print,calculo_recta(i[0],x_print),label=f"w:{np.round(i[0][0],4)} epochs=0")
        plt.plot(x_print,calculo_recta(i[1],x_print),label=f"w:{np.round(i[1][0],4)} epochs={epochs_encontrada[index]+1}")
        plt.plot(x_print,calculo_recta(i[2],x_print),label=f"w:{np.round(i[2][0],4)} epochs=10")
        plt.plot(x_print,calculo_recta(i[3],x_print),label=f"w:{np.round(i[3][0],4)} epochs={epochs_totales}")

        plt.legend(loc="best")
        plt.show()

def calculo_recta(w,x):
    x2=-(w[0][0]*x/w[0][1])+(w[0][2]/w[0][1])
    return x2

def crear_gif(w,x,caso):
    imagenes=[]
    print(f"Creando imágenes {caso}")
    for index,i in tqdm(enumerate(w)):
        plt.scatter(x[:,0],x[:,1],marker="o")
        x_print=np.linspace(-0.5,1.5,3)
   
        if caso=="and":
            plt.scatter(1,1,marker="o",color="red")
            plt.title(f'Caso AND {index}')
        if caso=="or":
            plt.scatter(0,0,marker="o",color="red")
            plt.title(f'Caso OR {index}')
        
        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.xlim(-0.5,1.5)
        plt.ylim(-0.5,1.5)

        plt.plot(x_print,calculo_recta(i,x_print))
        plt.savefig(f'{caso}{index}.png')
        imagenes.append(f'{caso}{index}.png')
        plt.close()

    with imageio.get_writer(f'{caso}gif.gif', mode='I') as writer:
        print(f"Creando GIF {caso} ")
        for imagen in tqdm(imagenes):
            image = imageio.imread(imagen)
            writer.append_data(image)

    #eliminamos       
    for filename in set(imagenes):
        os.remove(filename)


if __name__ == '__main__':
    main()
    