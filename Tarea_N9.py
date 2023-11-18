#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def DibujarFuncion(inicial,final,division,Funcion):         #Funcion que plotea la funcion exacta
    rango=np.linspace(inicial,final,division)               #Inicio el rango, dividiendo el intervalo para mejorar la grafica
    ylist=[]                                                #Lista dinamica para y
    for j in range(division):
        ylist.append(Funcion(rango[j]))                     #Llenamos la lista y

    return rango,ylist

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
#---------------   EDO  -------------------------------

def Euler(inicial,final,h,ecuacion,con1,con2):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=con2
    for i in range(1,n):
        ylist[i]=ylist[i-1]+ecuacion(xlist[i-1],ylist[i-1])*(h)     #Aplicacion

    return xlist,ylist

def Predictor(x,y,h,ecuacion):
    y0=y+ecuacion(x,y)*(h)
    return y0

def Heun(inicial,final,h,ecuacion,con1,con2):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)             #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=con2
    for i in range(1,n):
        y0=Predictor(xlist[i-1],ylist[i-1],h,ecuacion)
        ylist[i]=ylist[i-1]+((ecuacion(xlist[i-1],ylist[i-1])+ecuacion(xlist[i],y0))/(2))*(h)

    return xlist,ylist

def HeunIterado(inicial,final,h,ecuacion,condicion,error=1e-4):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=condicion
    for i in range(1,n):
        y0=Predictor(xlist[i-1],ylist[i-1],h,ecuacion)
        ylist[i]=ylist[i-1]+((ecuacion(xlist[i-1],ylist[i-1])+ecuacion(xlist[i],y0))/(2))*(h)
        c=0
        maxiter=100
        Ea=error+1
        while Ea>error and c<maxiter:
            y_old=ylist[i]
            valor=ylist[i-1]+((ecuacion(xlist[i-1],ylist[i-1])+ecuacion(xlist[i],ylist[i]))/(2))*(h)
            ylist[i]=valor
            Ea=abs((ylist[i]-y_old)/(ylist[i]))*100
            c=c+1

    return xlist, ylist

def HeunModificado(inicial,final,h,ecuacion,anterior,condicion,error):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=condicion
    for i in range(1,n):
        y0=anterior+ecuacion(xlist[i-1],ylist[i-1])*2*h
        ylist[i]=ylist[i-1]+((ecuacion(xlist[i-1],ylist[i-1])+ecuacion(xlist[i],y0))/(2))*(h)
        c=0
        maxiter=100
        Ea=error+1
        while Ea>error and c<maxiter:
            y_old=ylist[i]
            ylist[i]=ylist[i-1]+((ecuacion(xlist[i-1],ylist[i-1])+ecuacion(xlist[i],y_old))/(2))*(h)

            Ea=abs((ylist[i]-y_old)/(ylist[i]))*100
            c=c+1

        anterior=ylist[i-1]

    return xlist,ylist

def RK2(inicial,final,h,ecuacion,con1,con2,a2):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=con2
    #Rk2
    a1=1-a2
    p1=q11=(1)/(2*a2)
    for i in range(1,n):
        k1=ecuacion(xlist[i-1],ylist[i-1])
        k2=ecuacion(xlist[i-1]+p1*h,ylist[i-1]+q11*k1*h)

        ylist[i]=ylist[i-1]+(a1*(k1)+a2*(k2))*h

    return xlist,ylist

def RK3(inicial,final,h,ecuacion,con1,con2):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=con2
    for i in range(1,n):
        k1=ecuacion(xlist[i-1],ylist[i-1])
        k2=ecuacion(xlist[i-1]+(1/2)*h,ylist[i-1]+(1/2)*k1*h)
        k3=ecuacion(xlist[i-1]+h,ylist[i-1]-k1*(h)+2*k2*h)

        ylist[i]=ylist[i-1]+(1/6)*(k1+4*(k2)+k3)*h

    return xlist,ylist

def RK4(inicial,final,h,ecuacion,con1,con2):
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros(n,dtype=np.float64)
    ylist[0]=con2
    for i in range(1,n):
        k1=ecuacion(xlist[i-1],ylist[i-1])
        k2=ecuacion(xlist[i-1]+(1/2)*h,ylist[i-1]+(1/2)*k1*h)
        k3=ecuacion(xlist[i-1]+(1/2)*h,ylist[i-1]+(1/2)*k2*h)
        k4=ecuacion(xlist[i-1]+h,ylist[i-1]+k3*h)

        ylist[i]=ylist[i-1]+(1/6)*(k1+2*(k2)+2*(k3)+k4)*h

    return xlist,ylist



#   Para sistemas de edos

def EulerModificado(inicial,final,h,sistema,condiciones):   #Variable numero, el numero de la ecuacion
    c=condiciones.shape[0]                                  #Cantidad de condiciones y
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros((n,c),dtype=np.float64)                  #Matriz con los y de cada ecuacion
    for j in range(c):                                      #Pongo las condiciones
        ylist[0,j]=condiciones[j]

    for i in range(1,n):                                    #Uso el metodo
        for j in range(c):
            ylist[i,j]=ylist[i-1,j]+sistema(j,xlist[i-1],ylist[i-1,])*(h)

    return xlist,ylist

def RK5Modificado(inicial,final,h,sistema,condiciones):
    c=condiciones.shape[0]                                  #Cantidad de condiciones y
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros((n,c),dtype=np.float64)
    klist=np.zeros((6,c),dtype=np.float64)                  #Lista de los K

    for j in range(c):
        ylist[0,j]=condiciones[j]

    for i in range(1,n):                                    #Uso el metodo
        for j in range(c):
            klist[0,j]=sistema(j,xlist[i-1],ylist[i-1,])
        for j in range(c):
            klist[1,j]=sistema(j,xlist[i-1]+(1/4)*h,ylist[i-1,]+(1/4)*(klist[0,])*h)
        for j in range(c):
            klist[2,j]=sistema(j,xlist[i-1]+(1/4)*h,ylist[i-1,]+(1/8)*(klist[0,])*h+(1/8)*(klist[1,])*h)
        for j in range(c):
            klist[3,j]=sistema(j,xlist[i-1]+(1/2)*h,ylist[i-1,]-(1/2)*(klist[1,])*h+(klist[2,])*h)
        for j in range(c):
            klist[4,j]=sistema(j,xlist[i-1]+(3/4)*h,ylist[i-1,]+(3/16)*(klist[0,])*h+(9/16)*(klist[3,])*h)
        for j in range(c):
            klist[5,j]=sistema(j,xlist[i-1]+h,ylist[i-1,]-(3/7)*(klist[0,])*h+(2/7)*(klist[1,])*h+(12/7)*(klist[2,])*h-(12/7)*(klist[3,])*h+(8/7)*(klist[4,])*h)

        for j in range(c):
            ylist[i,j]=ylist[i-1,j]+(1/90)*(7*klist[0,j]+32*(klist[2,j])+12*(klist[3,j])+32*klist[4,j]+7*klist[5,j])*h

    return xlist,ylist

#   Stiffness   #

def EulerImplicitoModificado(inicial,final,h,condiciones,constantes):       #Sistemas
    c=condiciones.shape[0]                                  #Cantidad de condiciones Y
    xlist=np.arange(inicial,final+h,h,dtype=np.float64)     #Funcion que divide el intervalo en partes iguales
    n=xlist.shape[0]
    ylist=np.zeros((n,c),dtype=np.float64)                  #Matriz con los y de cada ecuacion

    A=np.eye(c,dtype=np.float64)                            #Matriz del sistema (Identidad)
    b=np.zeros((c,1),dtype=np.float64)                      #Matriz b del sistema

    for j in range(c):                                      #Pongo las condiciones
        ylist[0,j]=condiciones[j]

    for i in range(c):                                      #Creo la matriz del sistema
        for j in range(c):
            A[i,j]=A[i,j]-constantes[i,j]*h

    for i in range(1,n):                                    #Uso el metodo
        for j in range(c):
            b[j,0]=ylist[i-1,][j]

        A1=np.copy(A)                                       #Para no cambiar la matriz A
        x = eliminacion_gauss(A1,b)                            #Solucion de esa iteracion

        for j in range(c):
            ylist[i,j]=x[j,0]

    return xlist,ylist

###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

def funcion(x):                                             #Solucion analitica
    return np.e**((x**(4)/4)-1.5*(x))

def ecuacion(x,y):
    return y*(x**3)-1.5*(y)

x,y=Euler(0,2,0.5,ecuacion,0,1)
plt.plot(x,y,'go-',label='Euler')

x,y=RK2(0,2,0.5,ecuacion,0,1,1/2)       #Heun simple
plt.plot(x,y,'ro-',label='Heun Simple')

x,y=RK2(0,2,0.5,ecuacion,0,1,2/3)       #Ralston
plt.plot(x,y,'bo-',label='Raslton')

# x,y=RK3(0,2,0.5,ecuacion,0,1)           #RK3
# plt.plot(x,y,'yo-',label='RK3')

x,y=RK4(0,2,0.5,ecuacion,0,1)           #RK4
plt.plot(x,y,'mo-',label='RK4')

x,y=DibujarFuncion(0,2,100,funcion)
plt.plot(x,y,'k-',label='Función')

plt.title('Ejercicio 1')
plt.grid()
plt.legend()
plt.show()

###############################################################################
################################# Ejercicio 2 #################################
###############################################################################

def ecuacion1(t,y):
    x1=y[0]
    x2=y[1]
    return (999*x1)+(1999*x2)

def ecuacion2(t,y):
    x1=y[0]
    x2=y[1]
    return (-1000*x1)-2000*x2

def sistema(numero,t,y):
    if numero==0:
        return ecuacion1(t,y)
    elif numero==1:
        return ecuacion2(t,y)
    
h=0.001                                          #Si aumento 2 0, se solapan
condiciones=np.array([1,1],dtype=np.float64)    #x1,x2 iniciales

t,y=EulerModificado(0,0.2,h,sistema,condiciones)

#--------   Euler Implicito ---------
constantes=np.array([[999,1999],[-1000,-2000]])

t1,y1=EulerImplicitoModificado(0,0.2,h,condiciones,constantes)

#   Grafica de la funcion x1
plt.plot(t,y[:,0],'b--',label='Explicito')
plt.plot(t1,y1[:,0],'g-',label='Implícito')

plt.legend(loc='upper left')
plt.grid()
plt.ylim(-5,5)
#   Grafica de la funcion x2
plt.plot(t,y[:,1],'b--',label='Explicito')
plt.plot(t1,y1[:,1],'g-',label='Implícito ')

plt.title('Ejercicio 2')
plt.legend(loc='upper right')
plt.grid()
plt.ylim(-3,5)
plt.grid()
plt.show()

###############################################################################
################################# Ejercicio 3 #################################
###############################################################################

def ecuacion3(t,y):
    return -0.5*y+(np.e**(-t))

# def sistema3(numero,t,y):
#     if numero == 0:
#         return ecuacion3(t,y)

h=0.5
anterior=5.222138
condicion=4.143883

x,y=Heun(2,3,h,ecuacion3,2,condicion)
plt.plot(x,y,'ro--',label='Heun')

x,y=HeunModificado(2,3,h,ecuacion3,anterior,condicion,0.001)        # Funciona
plt.plot(x,y,'mo--',label='Heun Modificado')

# condiciones=np.array([4.143883],dtype=np.float64)
# x,y=RK5Modificado(2,3,h,sistema3,condiciones)        # Funciona
# plt.plot(x,y,'kx--',label='RK5')
plt.title("Ejercicio 3")
plt.xlabel('X',fontsize=11)
plt.ylabel('Y',fontsize=11)
plt.yscale('log')
plt.legend(loc='upper right')
plt.grid()

plt.show()
