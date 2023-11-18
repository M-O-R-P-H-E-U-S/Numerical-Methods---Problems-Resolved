#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def sustitucion_backward(A, b): #Usa la sustitucion backward. Admite matrices triangulares superiores    
    
    N = len(A)
    x=np.zeros((N,1),dtype=np.float64)
    for i in range(N-1,-1,-1):               
        suma=0
        for j in range (i+1,N):                 
            suma = suma + A[i,j]*x[j,0]
        x[i] = (b[i,0] - (suma))/A[i,i]     
        #print(x[i], i)
    return x 

def eliminacion_gauss(A,b):
    N = len(A)                                    
    x = np.zeros((N,1),np.float64)
    for k in range(N-1):
        p=k
        big=abs(A[k,k])                         
        for i in range(k+1,N):                  #Se busca en la columna de k,k
            dummy=abs(A[i,k])                   #Se guarda el valor buscando en la columna, para comparar
            if dummy>big:                       #Si encuentra un valor mas grande que el valor de la diagonal, lo cambia en la diagonal
                big=dummy
                p=i                             #Cambia la fila de la diagonal por esa

        if p!=k:                                #Si la p cambio entonces lleva toda la fila junto con ella
            for j in range(N):
                dummy=A[p,j]
                A[p,j]=A[k,j]
                A[k,j]=dummy

            dummy=b[p,0]
            b[p,0]=b[k,0]
            b[k,0]=dummy

        for i in range(k+1,N):                      
            factor=(A[i,k])/(A[k,k])                #Se recorre las filas partiendo de k+1
            for j in range(N):                      #Se recorre los indices de la colmunas para una fila i
                A[i,j]=(A[i,j])-(factor*(A[k,j]))   #Eliminación de Gauss elemento a elemento

            b[i,0]=(b[i,0])-(factor*b[k,0])         

        x = sustitucion_backward(A, b)

    return x

def InterLagrange(valor,D):                                 #Funcion que ejecuta la interpolacion de Lagrange para un valor dado
    f=len(D)                                                
    base=np.zeros((f,1),dtype=np.float64)                   #Se define la matriz columna base, donde irán las bases de Lagrange
    lagrange=np.zeros((f,1),dtype=np.float64)               #Se define la matriz de Lagrange, donde irán las bases multiplicadas con los elementos f(x)
    for i in range(f):
        prod1=1
        prod2=1
        for j in range(i):
            prod1=prod1*((valor-D[j,0])/(D[i,0]-D[j,0]))    #Producto de una base de Lagrange j<i
        for k in range(i+1,f):
            prod2=prod2*((valor-D[k,0])/(D[i,0]-D[k,0]))    #Producto de una base de Lagrange k>i
        base[i,0]=prod1*prod2                               #Array de bases de lagrange para un valor dado
    for l in range(f):
        lagrange[l,0]=base[l,0]*D[l,1]                      #Se genera un array con los elemento f(x)*l
    L=np.sum(lagrange)                                      #Se suman los elementos del array
    return L                                                #Valor encontrado por el metodo de lagrange

def trapecio(a,b):
    I = (b-a)*(f1(a)+f1(b))/2
    return I

def PuntosEquidistantes(inicial,final,puntos,f):          #Función que divide un intervalo en puntos y los evalua en una funcion
    D=np.zeros((puntos,2),dtype=np.float64)               #Defino el array de datos
    listax=np.linspace(inicial,final,puntos)              #Lista que divide el intervalo
    for j in range(puntos):
        D[j,0]=listax[j]                                  #Se llena la parte x de los datos con el intervalo dividido
    for k in range(puntos):
        D[k,1]=f(listax[k])                               #Se llena la parte y de los datos evaluando el intervalo en la función
    return D

def trapecio_compuesto_eq(a, b, n, f):
    suma = 0
    x = PuntosEquidistantes(a, b, n, f)
    i = 1
    #print(x)
    while i<n:
        suma = suma + (x[i,0]-x[i-1,0])*(x[i-1,1]+x[i,1])/2
        i=i+1

    return suma

def trapecio_compuesto_no_eq(t, v):
    size = len(t)
    suma = 0
    x=np.zeros((size,2),dtype=np.float64)
    for i in range(size):
        x[i,0] = t[i]
        x[i,1] = v[i]
    i = 1
    #print(x)
    while i<size:
        suma = suma + (x[i,0]-x[i-1,0])*(x[i-1,1]+x[i,1])/2
        i=i+1
    
    return suma

def Simpson13(a, b, f):
    suma = 0
    n=3
    x = PuntosEquidistantes(a, b, n, f)
    h=np.float64((b-a)/n)
    suma = 2*h*(x[0,1]+4*x[1,1]+x[2,1])/6 + suma
    
    return suma

def Simpson13C(a, b, f, n):     # n debe ser múltiplo de 2. n es el número de intervalos.
    x = np.linspace(a, b, n+1)          # Si se ingresa n impar, saldrá un valor errado.
    suma1 = 0
    suma2 = 0
    for i in range(1, n, 2):
        suma1 = suma1 + f(x[i])
    for j in range(2, n-1, 2):
        suma2 = suma2 + f(x[j])
    Integral = (b - a)*(f(a) + 4*suma1 + 2*suma2 + f(b))/(3*n)
    return Integral

def Simpson38(a, b, f):
    suma = 0
    n=4
    x = PuntosEquidistantes(a, b, n, f)
    h=np.float64((b-a)/n)
    #for i in range():
    suma = 3*h*(x[0,1]+3*(x[1,1]+x[2,1])+x[3,1])/8
    
    return suma

def Simpson38C(a, b, f, n):     # n debe ser múltiplo de 3. n es el número de intervalos.
    x = np.linspace(-2, 4, n+1)         # Si n no es múltiplo de 3, saldrá un valor errado.
    suma1 = 0
    suma2 = 0
    suma3 = 0
    for i in range(3, n-1, 3):
        suma1 = suma1 + f(x[i])
    for j in range(1, n, 3):
        suma2 = suma2 + f(x[j])
    for k in range(2, n, 3):
        suma3 = suma3 + f(x[k])
    Integral = 3*(b - a)*(f(a) + 2*suma1 + 3*(suma2 + suma3) + f(b))/(8*n)
    
    return Integral

###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

print("Ejercicio 1")

def f1(x):
    return (1-x-4*x**3+2*x**5)

S2 = trapecio(-2,4)
print("Regla del Trapecio: ", S2)

S32 = trapecio_compuesto_eq(-2,4, 2, f1)
S34 = trapecio_compuesto_eq(-2,4, 4, f1)
print("Regla del Trapecio Compuesto n=2: ", S32)
print("Regla del Trapecio Compuesto n=4: ", S34)

S13 = Simpson13C(-2,4, f1, 3)
S38 = Simpson38C(-2,4, f1, 4)

print("Regla del Simpson 1/3: ", S13)
print("Regla del Simpson 3/8: ", S38)

###############################################################################
################################# Ejercicio 2 #################################
###############################################################################

print("Ejercicio 2")

t_d = np.array([1, 2, 3.25 , 4.5, 6, 7, 8, 8.5, 9, 10])
v_d = np.array([5, 6, 5.5, 7, 8.5, 8, 6, 7, 7, 5])
S2 = trapecio_compuesto_no_eq(t_d, v_d)
print("Trapecio Compuesto No Equedistante: ", S2)

###############################################################################
################################# Ejercicio 3 #################################
###############################################################################

print("Ejercicio 3")

def Richardson(a, b, f, err, maxiter): 
    Error = err+1
    i = 1
    while err<Error and i<(maxiter+1):
        n1=i
        h2 = (b-a)/(2*n1)
        h1 = (b-a)/n1 
        Inte=trapecio_compuesto_eq(a, b, 2*n1, f)+(1/((h1/h2)**(2)-1))*(trapecio_compuesto_eq(a, b, 2*n1, f)-trapecio_compuesto_eq(a, b, n1, f))
        i = i+1
        Error = (trapecio_compuesto_eq(a, b, n1, f)-trapecio_compuesto_eq(a, b, 2*n1, f))/(1-(h1/h2)**(2))*100
    return Inte

def f2(x):
    return (np.exp(x)*np.sin(x))/(1+x**2)

a3 = 0
b3 = 2
err = 0.5
maxiter = 100
S3 = Richardson(a3, b3, f2, err, maxiter)
print("Extrapolacion de Richardson: ", S3)
