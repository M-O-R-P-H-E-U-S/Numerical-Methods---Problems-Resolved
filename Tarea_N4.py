#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def jacobi(A, b, x_0, tol, itera):
    N = len(A)
    A = np.array(A)                      #Se convierte la matriz A en un numpy array en caso no se haya declarado como tal 
    b = np.array(b)                      #Análogamente, numpy array para b
    error, n_it = [], []                 #Declaración de los arrays de error y número de iteraciones respectivamente
    n=0                                  #Inicializador del contador de iteraciones
    ea=tol+1                             #Declaración del valor de error inicial, se suma 1 para asegurarse que ea>err
    x=np.zeros((N,1),dtype=np.float64)   #declaración del array que contiene las soluciones
    while ea>tol and n<itera:         #Dos condiciones: tolerancia de error y no superar el error
        for i in range(N):
            sum1 = 0         
            sum2 = 0                       
            for j in range(i):                          #For de la sumatoria i<j
                sum1=sum1+A[i,j]*x_0[j,0]
            for k in range(i+1,N):                      #For de la sumatoria i>j
                sum2=sum2+A[i,k]*x_0[k,0]
            x[i,0]=(b[i,0]-sum2-sum1)/A[i,i]
            #print(x)
        ea = np.linalg.norm(x-x_0,np.inf)/np.linalg.norm(x,np.inf)                          #Se halla el error relativo
        x_0 = np.copy(x)
        error.append(ea)                                    #Se agrega el error al array de error
        n=n+1                                               #Conteo de iteración
        n_it.append(n)                                      #Se agrega la iteración al array de iteraciones    
    return x, n, error, n_it                                #Valores a retornar

def gauss_seidel(A, b, x_0, tol, itera):
    N = len(A)
    A = np.array(A)
    b = np.array(b)
    error, n_it = [], []
    n = 0
    ea = tol + 1
    x = np.zeros((N,1),dtype=np.float64)
    while ea > tol and n < itera:
        for i in range(N):
            sum1=0
            sum2=0
            for j in range(i):
                sum1=sum1+A[i,j]*x[j,0]             #Primera sumatoria con los x_r          (i>j)
            for k in range(i+1,N):
                sum2=sum2+A[i,k]*x_0[k,0]       #Segunda sumatoria con los x_iniciales. (i<j)
            x[i,0]=(b[i,0]-sum2-sum1)/A[i,i]  
            #print(x)
        ea = np.linalg.norm(x-x_0,np.inf)/np.linalg.norm(x,np.inf)   
        x_0=np.copy(x)
        error.append(ea)
        n=n+1
        n_it.append(n)
    
    return x, n, error, n_it

def SOR(A, b, x_0, w, tol, itera):
    N = len(A)
    A = np.array(A)
    b = np.array(b)
    error, n_it = [], []
    n=0
    ea = tol + 1
    x=np.zeros((N,1),dtype=np.float64)
    while ea > tol and n < itera:
        for i in range(N):
            sum1=0
            sum2=0
            sum3=0
            for j in range(i):
                sum1=sum1+A[i,j]*x_0[j,0]                   #Sumatoria (1-w) (i>j)
            for k in range(i+1,N):
                sum2=sum2+A[i,k]*x_0[k,0]                   #Sumatoria normal (i<j)
            for l in range(i):
                sum3=sum3+A[i,l]*x[l,0]                         #Sumatoria (w)  (i>j)

            x[i,0]=(b[i,0]-(1-w)*sum1-sum2-w*sum3)/A[i,i]      #Aplicacion del método
            #print(x)
        ea = np.linalg.norm(x-x_0,np.inf)/np.linalg.norm(x,np.inf)   
        error.append(ea)
        n=n+1
        n_it.append(n)
        x_0=np.copy(x)
    
    return x, n, error, n_it


###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

# Parte A

A_11 = np.array([[1, 2, 1], [0, -4, 1], [0, 0, -2]],dtype=np.float64) 

b_11 = np.array([[5], [2], [4]],dtype=np.float64) 

x_11 = np.zeros((len(A_11),1))                                  #Creación de la matriz solución

j_11 = jacobi(A_11, b_11, x_11, 1.e-6, 1000)

#print(j_11[2])

#print(j_11[0])
A_12 = np.array([[1, 2, 1], [0, -4, 1], [0, 0, -2]],dtype=np.float64) 

b_12 = np.array([[5], [2], [4]],dtype=np.float64) 

x_12 = np.zeros((len(A_12),1)) 

g12_s = gauss_seidel(A_12, b_12, x_12, 1.e-6, 1000)

#print(g12_s[2])

#print(g12_s[0])

A_13 = np.array([[1, 2, 1], [0, -4, 1], [0, 0, -2]],dtype=np.float64) 

b_13 = np.array([[5], [2], [4]],dtype=np.float64) 

x_13 = np.zeros((len(A_13),1)) 

sor_13 = SOR(A_13, b_13, x_13, 0.95, 1.e-6, 1000)

#print(sor_1[0])

plt.title("Error vs iteraciones, Parte 1A")
plt.plot(j_11[3], j_11[2], marker='o', linestyle="-", color="black", label='Jacobi')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
#plt.yscale('log')
plt.show()

plt.title("Error vs iteraciones, Parte 1A")
plt.plot(g12_s[3], g12_s[2], marker='o', linestyle="-", color="red", label='Gauss-Seidel')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
#plt.yscale('log')
plt.show()

plt.title("Error vs iteraciones, Parte 1A")
plt.plot(sor_13[3], sor_13[2], marker='o', linestyle="-", color="green", label='SOR')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
#plt.yscale('log')
plt.show()

# Parte B

A_2 = np.array([[2, 0, 0], [1, 4, 0], [4, 3, 3]],dtype=np.float64) 

b_2 = np.array([[4], [2], [5]],dtype=np.float64) 

x_2 = np.zeros((len(A_2),1))                                  #Creación de la matriz solución

j_2 = jacobi(A_2, b_2, x_2, 1.e-6, 1000)

#print(j_2[2])

#print(j_2[0])
A_2 = np.array([[2, 0, 0], [1, 4, 0], [4, 3, 3]],dtype=np.float64) 

b_2 = np.array([[4], [2], [5]],dtype=np.float64) 

x_2 = np.zeros((len(A_2),1))                                  #Creación de la matriz solución

g2_s = gauss_seidel(A_2, b_2, x_2, 1.e-6, 1000)

#print(g2_s[2])

#print(g2_s[0])

A_2 = np.array([[2, 0, 0], [1, 4, 0], [4, 3, 3]],dtype=np.float64) 

b_2 = np.array([[4], [2], [5]],dtype=np.float64) 

x_2 = np.zeros((len(A_2),1))                                  #Creación de la matriz solución 

sor_2 = SOR(A_2, b_2, x_2, 0.95, 1.e-6, 1000)

#print(sor_2[2])

#print(sor_2[0])

plt.title("Error vs iteraciones, Parte 1B")
plt.plot(j_2[3], j_2[2], marker='o', linestyle="-", color="black", label='Jacobi')
plt.plot(g2_s[3], g2_s[2], marker='o', linestyle="-", color="red", label='Gauss Seidel')
plt.plot(sor_2[3], sor_2[2], marker='o', linestyle="-", color="green", label='SOR')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
#plt.yscale('log')
plt.show()

###############################################################################
################################# Ejercicio 2 #################################
###############################################################################

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

A_1, b_1 = voltaje_n(5,0,N)

x_1 = np.zeros((len(A_1),1))                                  #Creación de la matriz solución 

v_jacobi = jacobi(A_1, b_1, x_1, 1e-6, 1000)

#print(v_jacobi)

A, b = voltaje_n(5,0,N)

x = np.zeros((len(A),1))                                  #Creación de la matriz solución 

v_gs = gauss_seidel(A, b, x, 1e-6, 1000)

#print(v_gs)

A, b = voltaje_n(5,0,N)

x = np.zeros((len(A),1))                                  #Creación de la matriz solución 

v_sor = SOR(A, b, x, 1.5, 1e-6, 1000)

#print(v_sor)

plt.title("Error vs iteraciones N = 6, Parte 2")
plt.plot(v_jacobi[3], v_jacobi[2], marker='o', linestyle="-", color="black", label='Jacobi')
plt.plot(v_gs[3], v_gs[2], marker='o', linestyle="-", color="red", label='Gauss Seidel')
plt.plot(v_sor[3], v_sor[2], marker='o', linestyle="-", color="green", label='SOR')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()

# Para N = 1000

N = 10

A, b = voltaje_n(5,0,N)

x = np.zeros((len(A),1))                                  #Creación de la matriz solución 

v_jacobi = jacobi(A, b, x, 1e-6, 1000)

#print(v_jacobi)

A, b = voltaje_n(5,0,N)

x = np.zeros((len(A),1))                                  #Creación de la matriz solución 

v_gs = gauss_seidel(A, b, x, 1e-6, 1000)

#print(v_gs)

A, b = voltaje_n(5,0,N)

x = np.zeros((len(A),1))                                  #Creación de la matriz solución 

v_sor = SOR(A, b, x, 1.5, 1e-6, 1000)

#print(v_sor)

plt.title("Error vs iteraciones N = 10, Parte 2")
plt.plot(v_jacobi[3], v_jacobi[2], marker='o', linestyle="-", color="black", label='Jacobi')
plt.plot(v_gs[3], v_gs[2], marker='o', linestyle="-", color="red", label='Gauss Seidel')
plt.plot(v_sor[3], v_sor[2], marker='o', linestyle="-", color="green", label='SOR')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()

###############################################################################
################################# Ejercicio 3 #################################
###############################################################################

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x = np.zeros((len(A),1)) 

x_jacobi = jacobi(A, b, x, 1e-4, 1000)

#print(x_jacobi)

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x = np.zeros((len(A),1)) 

x_gs = gauss_seidel(A, b, x, 1e-4, 1000)

#print(x_gs)

A = np.array([[30, -20, 0], [-20, 30, -10], [0, -10, 10]],dtype=np.float64) 

b = np.array([[9.8], [9.8], [9.8]],dtype=np.float64) 

x = np.zeros((len(A),1)) 

x_sor = SOR(A, b, x, 1.5, 1e-4, 1000)

#print(x_sor)

plt.title("Error vs iteraciones Parte 3")
plt.plot(x_jacobi[3], x_jacobi[2], marker='o', linestyle="-", color="black", label='Jacobi')
plt.plot(x_gs[3], x_gs[2], marker='o', linestyle="-", color="red", label='Gauss-Seidel')
plt.plot(x_sor[3], x_sor[2], marker='o', linestyle="-", color="green", label='SOR')
plt.ylabel('Error relativo')
plt.xlabel('#Iteraciones')  
plt.legend()
plt.grid()
plt.yscale('log')
plt.show()

