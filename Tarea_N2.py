#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def busqueda_incremental(f, x_l, dx, tol, n):
    i = 0      
    x_u = x_l + dx
    while f(x_l)*f(x_u) >= 0:  
        i = i + 1            
        x_l = x_u
        x_u = x_u + dx
        print("\t x_l \t \t x_u \t f(x_l) \t f(x_u)")
        print('{:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(x_l , x_u , f(x_l) ,f(x_u)))
        e_abs = abs(x_u - x_l)
    print(e_abs)    
    return x_u

def metodo_biseccion(f, x_l, x_u, tol, n):
    if f(x_u)*f(x_l) > 0:
        print("La funcion no cambio de signo en este intervalo")
        return None
    e_abs = x_u - x_l
    elist =[]
    ilist =[]
    i = 0    
    while (i < n) and (e_abs > tol):   
        x_r = (x_u + x_l)/2.0
        if f(x_l)*f(x_r) < 0:
            x_u = x_r
        elif f(x_l)*f(x_r) > 0:
            x_l = x_r
        else:
            return x_r
        print("\t x_l \t  \t x_r \t x_u \t \t f(x_l) \t f(x_r)\t f(x_u)")
        print('{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(x_l ,x_r , x_u , f(x_l) ,f(x_r) ,f(x_u)))
        e_abs = abs(x_u - x_l)
        i = i + 1
        elist.append(e_abs)
        ilist.append(i)
    plt.plot(ilist,elist, marker = "o", label = "Biseccion")
    plt.title('Biseccion')
    plt.xlabel("# Iteraciones")
    plt.ylabel("Error")
    print(e_abs)
    #plt.show()
    return x_r

def metodo_falsa_posicion(f, x_l, x_u, tol, n):
    if f(x_u)*f(x_l) > 0:
        print("La funcion no cambio de signo en este intervalo")
        return None
    e_abs = x_u - x_l
    elist =[]
    ilist =[]
    i = 0
    while (i < n) and (e_abs > tol):
        x_r = x_u - (f(x_u)*(x_l - x_u))/(f(x_l) - f(x_u))
        if f(x_l)*f(x_r) < 0:
            x_u = x_r
        elif f(x_l)*f(x_r) > 0:
            x_l = x_r
        else:
            return x_r
        print("\t x_l \t  \t x_r \t x_u \t \t f(x_l) \t f(x_r)\t f(x_u)")
        print('{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(x_l ,x_r , x_u , f(x_l) ,f(x_r) ,f(x_u)))
        e_abs = abs(x_u - x_l)
        i = i + 1 
        elist.append(e_abs)
        ilist.append(i)
    plt.plot(ilist,elist, marker = "o", label = "Falsa")
    plt.title('Falsa Posicion')
    plt.xlabel("# Iteraciones")
    plt.ylabel("Error")
    print(e_abs)
    #plt.show()
    return x_r
    
def metodo_punto_fijo(g, x_0, tol, n):
    i = 0
    x_ant = x_0   
    elist =[]
    ilist =[]
    e_abs = 1 #abs((x_0 - x_ant)/x_0) 
    while (i < n) and (e_abs > tol):    
        print("\t x_0 \t   f(x_0)")
        print('{:10.5f} {:10.5f}'.format(x_0 , g(x_0)))
        x_0 = g(x_0)
        e_abs = abs(x_0 - x_ant)
        x_ant = x_0
        i = i + 1
        elist.append(e_abs)
        ilist.append(i)
    plt.plot(ilist,elist, marker = "o", label = "Fijo")
    plt.title('Punto Fijo')
    plt.xlabel("# Iteraciones")
    plt.ylabel("Error")
    #plt.show()
    print(e_abs)
    return x_0

def metodo_newton_raphson(f, df, x_0, tol, n):    
    i = 0
    x_ant = x_0
    elist =[]
    ilist =[]
    e_abs = 1 #abs((x_0 - x_ant)/x_0)     
    while (i < n) and (e_abs > tol):    
        x_0 = x_0 - f(x_0)/df(x_0)
        e_abs = abs(x_0 - x_ant)
        x_ant = x_0
        print("\t x_0 \t   f(x_0) \t df(x_0)")
        print('{:10.5f} {:10.5f} {:10.5f}'.format(x_0 , f(x_0), df(x_0)))
        i = i + 1
        elist.append(e_abs)
        ilist.append(i)
    plt.plot(ilist,elist, marker = "o", label = "Newton")
    plt.title('Punto Fijo')
    plt.xlabel("# Iteraciones")
    plt.ylabel("Error")
    #plt.show()
    print(e_abs)
    return x_0

def metodo_secante(f, x_l, x_u, tol, n):
    i = 0
    e_abs = 1 #abs((x_0 - x_ant)/x_0) 
    elist =[]
    ilist =[]
    while (i < n) and (e_abs > tol):    
        x = x_u - f(x_u)*(x_l - x_u)/(f(x_l) - f(x_u))
        x_l = x_u
        x_u = x
        e_abs = abs(x_u - x_l)
        print("\t x_l \t  \t x \t \t x_u \t f(x_l) \t f(x)\t f(x_u)")
        print('{:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f} {:10.5f}'.format(x_l ,x , x_u , f(x_l) ,f(x) ,f(x_u)))
        i = i + 1
        elist.append(e_abs)
        ilist.append(i)
    plt.plot(ilist,elist, marker = "o", label = "Secante")
    plt.legend()
    plt.title('Punto Fijo')
    plt.xlabel("# Iteraciones")
    plt.xlim(0,8)
    plt.ylabel("Error")
    plt.grid()
    plt.show()
    print(e_abs)
    return x
###############################################################################
################################# Ejercicio (b) ###############################
###############################################################################
def f(x):
    return 5*(np.exp(-x)) + x - 5

def df(x):
    return 1 - 5*(np.exp(-x))

def b(x):
    kB = 1.380649*(10**(-23))
    c = 3*10**(8)
    h = 6.62607015*10**(-34)
    return h*c/(kB*x)

def g(x):
    return 5 - 5*(np.exp(-x))

x = np.linspace(0.1, 100, 10000)

plt.plot(x,f(x))
plt.xlim(4, 6)
plt.ylim(-1,1)
plt.grid()
plt.show()

x_l = 4
x_u = 6
tol = 1e-6
n = 50

dx = 0.01
x_0 = 4

sol_busque = busqueda_incremental(f, x_l, dx, tol, n)

print("Solucion Busque", b(sol_busque))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_busque, b(sol_busque) ))
###############################################################################
sol_bi = metodo_biseccion(f, x_l, x_u, tol, n)

print("Solucion Biseccion", b(sol_bi))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_bi, b(sol_bi) ))
###############################################################################
sol_falsa = metodo_falsa_posicion(f, x_l, x_u, tol, n)

print("Solucion Falsa Posicion", b(sol_falsa))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_falsa, b(sol_falsa) ))
###############################################################################
sol_fijo = metodo_punto_fijo(g, x_0, tol, n)

print("Solucion Punto Fijo", b(sol_fijo))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_fijo, b(sol_fijo) ))
###############################################################################
sol_newton = metodo_newton_raphson(f, df, x_0, tol, n)

print("Solucion Newton Raphson", b(sol_newton))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_newton, b(sol_newton) ))
###############################################################################
sol_secante = metodo_secante(f, x_l, x_u, tol, n)

print("Solucion Secante", b(sol_secante))
print('Raíz \t |b')
print('{:.8}|{:.8}'.format(sol_secante, b(sol_secante) ))