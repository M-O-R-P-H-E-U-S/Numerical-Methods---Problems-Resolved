#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def sustitucion_forward(A, b): #Usa la sustitucion Forward. Solo admite Matrices triangulares inferiores
    
    N = len(A)
    x = np.zeros((N,1),dtype=np.float64)  
    for i in range(N):
        suma=0
        for j in range (i):                     
            suma = suma + A[i,j]*x[j,0]
        x[i]=(b[i,0] - (suma))/A[i,i] 
        
    return x

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

def pivoteo_parcial(A, b):
    N = len(b)                              
    for j in range(N):                      # j recorre las n columnas.
        if A[j, j] == 0:                    # Si un elemento de la digaonal es 0, entra en este bucle.
            i = np.argmax(np.abs(A[:,j]))   # i será el máximo valor absoluto de la columna j.
            A[[i, j], :] = A[[j, i], :]     # Intercambiamos, en A, la fila i, por la fila j con el 0 en la diagonal.
            b[j] = b[i]                     # Intercambiamos, en b, la fila i, por la fila j con el 0 en la diagonal.
    return A, b                             # Retorna A y b

def LU(A,b):                                
    N = len(A)                                
    L = np.zeros((N,N),dtype=np.float64)      #Se define la matriz L, inferior
    U = np.zeros((N,N),dtype=np.float64)      #Se define la matriz U, superior
    d = np.zeros((N,1),dtype=np.float64)      #Se define la matriz d
    x = np.zeros((N,1),dtype=np.float64)      
    for j in range(N):                      
        L[j,j]=1                            #Se inicializa la diagonal de L con unos
        for i in range (0,j+1):
            suma=0
            for k in range(i):              #Sumatoria de U
                suma=suma+L[i,k]*U[k,j]
            U[i,j]=(A[i,j]-suma)
        for i in range(j+1,N):
            suma=0
            for k in range(j):              #Sumatoria de L
                suma=suma+(L[i,k]*U[k,j])
            L[i,j]=(A[i,j]-suma)/(U[j,j])

    d = sustitucion_forward(L, b)                      
    x = sustitucion_backward(U, d)                     
    
    return x

def cholesky(A,b):
    
    N = len(A)                               
    G=np.zeros((N,N),dtype=np.float64)      #Se define la matriz G de Cholesky
    d=np.zeros((N,1),dtype=np.float64)      #Se define la matriz d
    x=np.zeros((N,1),dtype=np.float64)      
    for i in range(N):
        suma=0
        for k in range(i):                  #El for de la sumatoria
            suma=suma+((G[k,i])**2)
        G[i,i]=np.sqrt(A[i,i]-suma)         #Almacenamiento del resultado en G[i,i]

        for j in range(i+1,N):              #2da sumatoria, índice j
            suma=0
            for k in range(i):              #2da sumatoria, índice k
                suma=suma+G[k,i]*G[k,j]
            G[i,j]=((A[i,j]-suma)/(G[i,i])) 

    d = sustitucion_forward(np.transpose(G),b)                    #Cálculo la matriz d
    x = sustitucion_backward(G,d)                     
    
    return x

###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

A_1 = np.array([[1, 2, 1], [0, -4, 1], [0, 0, -2]],dtype=np.float64) 

b_1 = np.array([[5], [2], [4]],dtype=np.float64) 

#
x_11 = sustitucion_backward(A_1, b_1)

print(x_11)

A_2 = np.array([[2, 0, 0], [1, 4, 0], [4, 3, 3]],dtype=np.float64) 

b_2 = np.array([[4], [2], [5]],dtype=np.float64) 

x_2 = sustitucion_forward(A_2, b_2)

print(x_2)



###############################################################################
################################# Ejercicio 2 #################################
###############################################################################
'''
def voltaje_n(v_i,v_f,N):
    A=np.zeros((N,N))
    B=np.zeros((N,1))

    e_2=[3,-1,-1,3]
    b_2=[v_i + v_f,v_i + v_f]
    e_3=[3,-1,-1,-1,4,-1,-1,-1,3]
    b_3=[v_i,v_i + v_f,v_f]
    e_4=[3,-1,-1,0,-1,4,-1,-1,-1,-1,4,-1,0,-1,-1,3]
    b_4=[v_i,v_i,v_f,v_f]

    for i in range(N):
        for j in range(N):
            if N==1:
                A[i,j]=2
                B[i,0]=v_i + v_f
            elif (N==2):
                A[i,j]=e_2[j+i*N]
                B[i,0]=b_2[i]
            elif (N==3):
                A[i,j]=e_3[j+i*N]
                B[i,0]=b_3[i]
            elif (N==4):
                A[i,j]=e_4[j+i*N]
                B[i,0]=b_4[i]
            else:
                if 1<i and i<N-2:
                    A[i,i-2],A[i,i-1],A[i,i],A[i,i+1],A[i,i+2]=-1,-1,4,-1,-1
                    B[i,0]=0

                A[0,0],A[0,1],A[0,2]=3,-1,-1
                B[0,0]=v_i
                A[1,0],A[1,1],A[1,2],A[1,3]=-1,4,-1,-1
                B[1,0]=v_i
                A[N-2,N-4],A[N-2,N-3],A[N-2,N-2],A[N-2,N-1]=-1,-1,4,-1
                B[N-2,0]=v_f
                A[N-1,N-3],A[N-1,N-2],A[N-1,N-1]=-1,-1,3
                B[N-1,0]=v_f

    return A, B

# Para N = 6

N = 6

A, b = voltaje_n(5,0,N)

print(A)

#print(b)

v_gauss = eliminacion_gauss(A, b)

print(v_gauss)

A, b = voltaje_n(5,0,N)

v_LU = LU(A, b)

print(v_LU)

A, b = voltaje_n(5,0,N)

v_cholesky = cholesky(A, b)

print(v_cholesky)

# Para N = 1000

N = 200

A, b = voltaje_n(5,0,N)

#print(A)

#print(b)

v_gauss = eliminacion_gauss(A, b)

print(v_gauss)

A, b = voltaje_n(5,0,N)

v_LU = LU(A, b)

print(v_LU)

A, b = voltaje_n(5,0,N)

v_cholesky = cholesky(A, b)

print(v_cholesky)

###############################################################################
################################# Ejercicio 3 #################################
###############################################################################

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x_gauss = eliminacion_gauss(A,b)

print(x_gauss)

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x_LU = LU(A, b)

print(x_LU)

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x_cholesky = cholesky(A,b)

print(x_cholesky)
'''