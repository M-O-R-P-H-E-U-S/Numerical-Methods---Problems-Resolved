#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings
import math

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

def regresion_lineal(x, y, xx):
    n1 = len(x)
    n2 = len(y)
    sum_x, sum_y, sum_xy, sum_x2, st, sr = 0, 0, 0, 0, 0, 0
    if n1==n2:
        for i in range(n1):
            sum_x = sum_x + x[i]
            sum_y = sum_y + y[i]
            sum_xy = sum_xy + x[i]*y[i]
            sum_x2 = sum_x2 + x[i]*x[i]
        xm = sum_x/n1
        ym = sum_y/n1
        a1 = (n1*sum_xy - sum_x*sum_y)/(n1*sum_x2 - sum_x**2)
        a0 = ym - a1*xm
        yy = a1*xx + a0
        for i in range(n1):
            st = st + (y[i] - ym)**2
            sr = sr + (y[i] - a0 - a1*x[i] )**2
        syx = math.sqrt(st/(n1-2))
        r2 = (st-sr)/st
    else:
        print("Verificar longitudes de x e y")

    return yy, a0, a1, r2, syx

def Regresion_Lineal_Multiple(X1,X2,X3,Y,x1,x2,x3):
	n = len(X1)
	A = np.zeros((4,4), dtype=np.float64)
	b = np.zeros(4, dtype=np.float64)
	X0 = np.ones(n)
	X = [ X0,X1,X2,X3 ]
	for i in range(4):
		for j in range(i,4):
			A[i,j] = sum(X[i]*X[j])
			A[j,i] = A[i,j]
		b[i] = sum(X[i]*Y)
	a0, a1, a2, a3 = LU(A,b)
	st = sum( (Y- sum(Y)/n)**2 )
	sr = sum( (Y -a0 -a1*X1 -a2*X2 - a3*X3)**2 )
	r = np.sqrt((st-sr)/st)
    
	return [a0,a1,a2,a3,r], a0 + a1*x1 + a2*x2 + a3*x3

def dfda0(x, a1):
    return x*np.exp(a1*x)

def dfda1(x, a0, a1):
    return a0*(x**2)*np.exp(a1*x)

def regress_no_lineal(x, y, a0, a1, err, maxiter):
    
    size = len(x)

    Ea=err+1
    i=0
    
    #a_old = np.zeros((2,1),dtype=np.float64)
    a_new = np.zeros((2,1),dtype=np.float64)
    a_new[0] = a0
    a_new[1] = a1
    
    while Ea>err and i<maxiter:
        
        D = np.zeros((size,1),dtype=np.float64)
        
        Z = np.zeros((size,2),dtype=np.float64)
        
        for i in range(size):
            D[i] = y[i] - f2(x[i], a0, a1)
                
        for i in range(size):
            Z[i,0] = dfda0(x[i], a1)
        for i in range(size):
            Z[i,1] = dfda1(x[i], a0, a1)    
        
        Z_T = np.transpose(Z)
        Z_TZ = np.dot(Z_T, Z)
        Z_TD = np.dot(Z_T, D)
        
        Sol = eliminacion_gauss(Z_TZ,Z_TD)
        
        a_old = Sol + a_new

        i = i + 1
        Ea=np.linalg.norm(a_old-a_new)/np.linalg.norm(a_old)     
        
        a_new = np.copy(a_old)
        
        a0 = a_new[0] 
        a1 = a_new[1]    
        
    a0 = float(a_old[0])
    a1 = float(a_old[1])
    
    return a0, a1

###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

x = np.array([0.5, 1, 2, 3, 4 ],dtype=np.float64) 

y = np.array([10.4, 5.8, 3.3, 2.4, 2 ],dtype=np.float64) 

plt.scatter(x, y, color = 'red', label = "real")
plt.grid()

xx = np.linspace(0.4,4.1,100)

x1 = 1/np.sqrt(x)

y1 = np.sqrt(y)

y11 = regresion_lineal(x1, y1, xx)

def cambio(a0, a1):
    b = 1/a0
    a = a1*b
    return a, b

def f1(x, a, b):
    return ((a+np.sqrt(x))/(b*np.sqrt(x)))**2

CV = cambio(y11[1], y11[2])

print("Problema 1")
print("Valor de a: ", CV[0])
print("Valor de b: ", CV[1])

plt.plot(xx, f1(xx, CV[0], CV[1]), color = "black", label = 'regresion-lineal')
plt.title("Ejercicio 1 - Curva")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

xx1 = np.linspace(0.4,2,100)

def lineal(x, a0, a1):
    return x*a1 + a0

plt.scatter(x1, y1, color = "red", label = 'datos')
plt.plot(xx1,lineal(xx1,y11[1], y11[2] ) ,color = "black", label = 'regresion-lineal')
plt.title("Ejercicio 1 - Linealizado")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

###############################################################################
################################# Ejercicio 2 #################################
###############################################################################

# Parte 1

x2 = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8],dtype=np.float64) 

y2 = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18],dtype=np.float64) 

plt.scatter(x2, y2, color = 'red', label = "real")
plt.grid()

xx2 = np.linspace(0, 1.9, 100)

y22 = np.log(y2/x2)

y22 = regresion_lineal(x2, y22, xx2)

def cambio2(a0, a1):
    a = np.exp(a0)
    b = a1
    return a, b
 
def f2(x, alpha, betha):
    return alpha*x*np.exp(betha*x)

CV2 = cambio2(y22[1], y22[2])
print("Problema 2 - Parte 1")
print("Valor de a: ", CV2[0])
print("Valor de b: ", CV2[1])

plt.plot(xx2, f2(xx2, CV2[0], CV2[1]), color = "black", label = 'regresion-lineal')
plt.title("Ejercicio 2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Parte 2

print("Problema 2 - Parte 2")

x = np.linspace(0, 1.9, 100)

x2 = np.array([0.1, 0.2, 0.4, 0.6, 0.9, 1.3, 1.5, 1.7, 1.8])
y2 = np.array([0.75, 1.25, 1.45, 1.25, 0.85, 0.55, 0.35, 0.28, 0.18])

a0_i = 1
a1_i = 1
err = 10**-6
maxiter = 100

S_N = regress_no_lineal(x2, y2, a0_i, a1_i, err, maxiter)

print("Valor de a: ", S_N[0])
print("Valor de b: ", S_N[1])

plt.scatter(x2, y2, color = 'red', label = "real")
plt.plot(xx2, f2(xx2, CV2[0], CV2[1]), color = "blue", label = 'regresion-lineal')
plt.plot(xx2, f2(xx2, S_N[0], S_N[1]), color = "green", label = 'regresion-nolineal')
plt.title("Ejercicio 2")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()


