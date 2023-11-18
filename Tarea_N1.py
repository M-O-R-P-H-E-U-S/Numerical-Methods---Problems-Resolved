import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


#######################  PARTE 1 ##############################################

r = 1

x = np.zeros([3, 1000])

x[0, 0] = 0.45

x[1,0] = 0.5

x[2,0] = 0.55

for i in range(len(x[0,:])):
    if i != 999:
          
        x[0,i+1] = r*(x[0,i])*(1-x[0,i])
            
        x[1,i+1] = r*(x[1,i])*(1-x[1,i])
            
        x[2,i+1] = r*(x[2,i])*(1-x[2,i])

plt.subplot(1, 3, 1)
plt.plot(x[0, :], "-o", color = "black")
plt.xlabel("# Iteraciones")
plt.ylabel("x_k")
plt.title("x0 = 0.45")

plt.grid()

plt.subplot(1, 3, 2)
plt.plot(x[1, :], "-o", color = "blue")
plt.xlabel("# Iteraciones")
plt.title("x0 = 0.50")

plt.grid()

plt.subplot(1, 3, 3)
plt.plot(x[2, :], "-o", color="green")
plt.xlabel("# Iteraciones")
plt.title("x0 = 0.55")
plt.grid()

plt.show()


#######################  PARTE 2 ##############################################

r = np.arange(1,4.1,0.1,dtype = np.float64)

x = np.zeros([len(r),3000],dtype = np.float64)
print(len(r))
for j in range(len(r)):
    x[j,0] = 0.5

    for i in range(0, 2999):
        
        x[j,i+1] = float(r[j])*float((x[j,i])*(1-x[j,i]))
    
plt.plot(r[:], x[:, 1000:3000],marker = "o", color = "black",markersize=0)    
plt.xlabel("r")
plt.ylabel("x")
plt.grid()
plt.xlim(0.9, 4.1)
plt.show()

#######################  PARTE 3 ##############################################

N = 100

itera = 100

x = np.linspace(-2, 2, N)

y = np.linspace(-2, 2, N)

z = np.zeros((len(x), len(y)), dtype = complex)

print(x[0],x[-1])

for i in range(N):
        
    for j in range(N):
            
        c = complex(x[i], y[j])
        zn = complex(0, 0)
        
        k = 0
        
        while (np.abs(zn) <= 2) and (k < itera): 
                
            zn = (zn)**2 + c
            
            k = k + 1
        
        if np.abs(zn) <= 2:
            
            plt.scatter(x[i], y[j], marker = "o", color="black")
        
plt.title("Conjunto de Maldelbrot")    
plt.ylabel("y")                
plt.xlabel("x")
plt.grid()
plt.show()
