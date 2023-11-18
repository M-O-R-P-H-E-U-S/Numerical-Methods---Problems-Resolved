#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def f(x):
    return (4/(1+x**(2)))


def CuadraturaGaussLegendre(f, a, b, n):   # n es el número de puntos.

    if n == 1:
        x = np.array([0],dtype=np.float64)
        w = np.array([2],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 2:
        x = np.array([-1/np.sqrt(3), 1/np.sqrt(3)],dtype=np.float64)
        w = np.array([1, 1],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 3:
        x = np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)],dtype=np.float64)
        w = np.array([5/9, 8/9, 5/9],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))
            
    if n == 4:
        x = np.array([-np.sqrt(3/7 + (2/7)*np.sqrt(6/5)), -np.sqrt(3/7 - (2/7)*np.sqrt(6/5)), np.sqrt(3/7 - (2/7)*np.sqrt(6/5)), np.sqrt(3/7 + (2/7)*np.sqrt(6/5))],dtype=np.float64)
        w = np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 5:
        x = np.array([-(np.sqrt(5 + 2*np.sqrt(10/7)))/3, -(np.sqrt(5 - 2*np.sqrt(10/7)))/3, 0, (np.sqrt(5 - 2*np.sqrt(10/7)))/3, (np.sqrt(5 + 2*np.sqrt(10/7)))/3],dtype=np.float64)
        w = np.array([(322 - 13*np.sqrt(70))/900, (322 + 13*np.sqrt(70))/900, 128/225, (322 + 13*np.sqrt(70))/900, (322 - 13*np.sqrt(70))/900],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))
    
    Integral = ((b-a)/2)*Integral
    
    return Integral

def CuadraturaGaussRadauLegendre(f, a, b, n):
    
    if n == 3:
        x = np.array([-1.000000, -0.289898, 0.689898],dtype=np.float64)
        w = np.array([0.222222, 1.0249717, 0.7528061],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 4:
        x = np.array([-1.000000, -0.575319, 0.181066, 0.822824],dtype=np.float64)
        w = np.array([0.125000, 0.657689, 0.776387, 0.440924],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 5:
        x = np.array([-1.000000, -0.720480, -0.167181, 0.446314, 0.885792],dtype=np.float64)
        w = np.array([0.080000, 0.446208, 0.623653, 0.562712, 0.287427],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))
    
    Integral = ((b-a)/2)*Integral
    
    return Integral

def CuadraturaGaussLobattoLegendre(f, a, b, n):
    if n == 2:
        x = np.array([-1.0, 1.0],dtype=np.float64)
        w = np.array([1.0, 1.0],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 3:
        x = np.array([-1.0, 0.0, 1.0])
        w = np.array([0.33, 1.33, 0.33])
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 4:
        x = np.array([-1.0, -0.447213595499958, 0.447213595499958, 1],dtype=np.float64)
        w = np.array([0.166666666666667, 0.833333333333333, 0.833333333333333,0.166666666666667],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 5:
        x = np.array([-1.0, -0.654653670707977, 0.0, 0.654653670707977, 1.0],dtype=np.float64)
        w = np.array([0.10, 0.544444444444444, 0.711111111111111, 0.544444444444444, 0.10],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 6:
        x = np.array([-1.0, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1.0],dtype=np.float64)
        w = np.array([0.066666666666667, 0.378474956297847, 0.5548588377035486, 0.5548588377035486, 0.378474956297847, 0.066666666666667],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))

    if n == 7:
        x = np.array([-1.0, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1.0],dtype=np.float64)
        w = np.array([0.047619047619048, 0.276826047361566, 0.431745381209863, 0.487619047619048, 0.431745381209863, 0.276826047361566, 0.047619047619048],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))
    
    if n == 8:
        x = np.array([-1.0, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.209299217902479, 0.591700181433142, 0.871740148509607, 1.0],dtype=np.float64)
        w = np.array([0.035714285714286, 0.210704227143506, 0.341122692483504, 0.412458794658704, 0.412458794658704, 0.341122692483504, 0.210704227143506, 0.035714285714286],dtype=np.float64)
        Integral = 0
        for i in range(n):
            Integral  = Integral + w[i]*(f(((b-a)/2)*x[i]+((a+b)/2)))
            
    Integral = ((b-a)/2)*Integral
    
    return Integral


def DiferenciacionForwardOh1(x, y, i):      # Aquí debemos ingresar x e y como arrays e i como el punto donde la función será derivada.
    Df1 = (y[i+1] - y[i])/(x[i+1] - x[i])
    Df2 = (y[i+2] - 2*y[i+1] + y[i])/(x[i+1] - x[i])**2
    Df3 = (y[i+3] - 3*y[i+2] + 3*y[i+1] - y[i])/(x[i+1] - x[i])**3  # Comentar esta linea si se tiene 3 o menos datos.
    Df4 = (y[i+4] - 4*y[i+3] + 6*y[i+2] - 4*y[i+1] + y[i])/(x[i+1] - x[i])**4   # Comentar esta linea si se tiene 4 o menos datos.
    return Df1, Df2, Df3, Df4

def DiferenciacionForwardOh2(x, y, i):
    Df1 = (-y[i+2] + 4*y[i+1] - 3*y[i])/(2*(x[i+1] - x[i]))
    Df2 = (-y[i+3] + 4*y[i+2] - 5*y[i+1] + 2*y[i])/((x[i+1] - x[i])**2)
    # Df3 = (-3*y[i+4] + 14*y[i+3] - 24*y[i+2] + 18*y[i+1] - 5*y[i])/(2*(x[i+1] - x[i])**3)
    # Df4 = (-2*y[i+5] + 11*y[i+4] - 24*y[i+3] + 26*y[i+2] - 14*y[i+1] + 3*y[i])/((x[i+1] - x[i])**4)
    return Df1, Df2

def DiferenciacionBackwardOh1(x, y, i):
    Df1 = (y[i] - y[i-1])/(x[i] - x[i-1])
    Df2 = (y[i] - 2*y[i-1] + y[i-2])/(x[i] - x[i-1])**2
    Df3 = (y[i] - 3*y[i-1] + 3*y[i-2] - y[i-3])/(x[i] - x[i-1])**3  # Comentar esta linea si se tiene 3 o menos datos.
    Df4 = (y[i] - 4*y[i-1] + 6*y[i-2] - 4*y[i-3] + y[i-4])/(x[i] - x[i-1])**4   # Comentar esta linea si se tiene 4 o menos datos.
    return Df1, Df2, Df3, Df4

def DiferenciacionBackwardOh2(x, y, i):
    Df1 = (3*y[i] - 4*y[i-1] + y[i-2])/(2*(x[i] - x[i-1]))
    Df2 = (2*y[i] - 5*y[i-1] + 4*y[i-2] - y[i-3])/(x[i] - x[i-1])**2
    # Df3 = (5*y[i] - 18*y[i-1] + 24*y[i-2] - 14*y[i-3] + 3*y[i-4])/(2*(x[i] - x[i-1])**3)
    # Df4 = (3*y[i] - 14*y[i-1] + 26*y[i-2] - 24*y[i-3] + 11*y[i-4] - 2*y[i-5])/(x[i] - x[i-1])**4
    return Df1, Df2

def DiferenciacionCentradaOh2(x, y, i):
    Df1 = (y[i+1] - y[i-1])/(2*(x[i+1] - x[i]))
    Df2 = (y[i+1] - 2*y[i] + y[i-1])/(x[i+1] - x[i])**2
    # Df3 = (y[i+2] - 2*y[i+1] + 2*y[i-1] - y[i-2])/(2*((x[i+1] - x[i])**3))
    # Df4 = (y[i+2] - 4*y[i+1] + 6*y[i] - 4*y[i-1] + y[i-2])/(x[i+1] - x[i])**4
    return Df1, Df2

def DiferenciacionCentradaOh4(x, y, i):
    Df1 = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2])/(12*(x[i+1] - x[i]))
    Df2 = (-y[i+2] + 16*y[i+1] - 30*y[i] + 16*y[i-1] - y[i-2])/(12*((x[i+1] - x[i])**2))
    Df3 = (-y[i+3] + 8*y[i+2] - 13*y[i+1] + 13*y[i-1] - 8*y[i-2] + y[i-3])/(8*((x[i+1] - x[i])**3)) # Comentar esta linea si se tiene 5 o menos datos.
    Df4 = (-y[i+3] + 12*y[i+2] + 39*y[i+1] + 56*y[i] - 39*y[i-1] + 12*y[i-2] + y[i-3])/(6*((x[i+1] - x[i])**4)) # Comentar esta linea si se tiene 6 o menos datos.
    return Df1, Df2, Df3, Df4


###############################################################################
################################# Ejercicio 1 #################################
###############################################################################

a, b = 0, 1 

print('Valor real de pi: ', np.pi)

I = CuadraturaGaussLegendre(f, a, b, 3)

print("\n Cuadratura Gauss Legendre")

print('La integral para 3 puntos en GL es:')
print(I)

I = CuadraturaGaussLegendre(f, a, b, 4)
print('La integral para 4 puntos en GL es:')
print(I)

I = CuadraturaGaussLegendre(f, a, b, 5)
print('La integral para 5 puntos en GL es:')
print(I)

###############################################################################

I = CuadraturaGaussRadauLegendre(f, a, b, 3)

print("\n Cuadratura Gauss RadauLegendre")

print('La integral para 3 puntos en GRL es:')
print(I)

I = CuadraturaGaussRadauLegendre(f, a, b, 4)
print('La integral para 4 puntos en GRL es:')
print(I)

I = CuadraturaGaussRadauLegendre(f, a, b, 5)
print('La integral para 5 puntos en GRL es:')
print(I)

###############################################################################

I = CuadraturaGaussLobattoLegendre(f, a, b, 3)

print("\n Cuadratura Gauss Lobatto Legendre")

print('La integral para 3 puntos en GLL es:')
print(I)

I = CuadraturaGaussLobattoLegendre(f, a, b, 4)
print('La integral para 4 puntos en GLL es:')
print(I)

I = CuadraturaGaussLobattoLegendre(f, a, b, 5)
print('La integral para 5 puntos en GLL es:')
print(I)

###############################################################################
################################# Ejercicio 2 #################################
###############################################################################


r = np.array([[200,5120],[202,5370],[204,5560],[206,5800],[208,6030],[210,6240]],dtype=np.float64)

theta = np.array([[200,0.75],[202,0.72],[204,0.70],[206,0.68],[208,0.67],[210,0.66]],dtype=np.float64)

























