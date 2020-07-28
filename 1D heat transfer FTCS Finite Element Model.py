import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import math


### Constants

rho = 980 # kg/m^3
k = 0.62 # W/(m*K)
cP = 4030 #J/(Kg*K)
alpha = k/(rho*cP)*(1000**2) # mm^2/sec
radius = 25 # mm 
h = 1168 # W/(m^2 K)
time_max = 13 # sec
T_ref = 52 # deg C
z = 7.8 # deg C
D = 4.5 # min
### FTCS Model Parameters

delta_r = 0.05 # mm
delta_t = 0.005 # sec
Fo = alpha*delta_t/(delta_r**2) # Fourier
Bi = h*(delta_r/1000)/k # Biot
stability = Fo*(Bi+1)


### Set up array

n_rows = math.ceil(time_max/delta_t) + 1
n_cols = math.ceil(radius/delta_r) + 1

print("number of timesteps: " + str(n_rows))
print("number of radius increments: " + str(n_cols))
print("stable solution: " + str(stability <= 0.5))


time_index = np.arange(0, time_max+delta_t, delta_t)
radius_index = np.arange(0, radius+delta_r, delta_r)[::-1]



## Apply convection BC at col 0 (r = 25 mm) and Evaluate Model for 3 temperatures
Tlist = [60, 80, 90] # degC
Solutions = list()
for T_inf in Tlist:
    T = np.zeros((n_rows, n_cols))
    T += 20 # Initial Condition
    for row in range(1, len(T)):
        T[row][0] = Fo*(2*Bi*T_inf+2*T[row-1][1]+(1/Fo-2*Bi-2)*T[row-1][0])
        for col in range(1, n_cols-1):
            T[row][col] = Fo*(T[row-1][col-1]+T[row-1][col+1]-2*T[row-1][col])+T[row-1][col]
        T[row][n_cols-1] = Fo*(2*T[row-1][col-1]-2*T[row-1][col])+T[row-1][col]
    Solutions.append(T)

### F calc - note: computed every 0.5 mm step independent of delta_r

def Fcalc_integrand(T):
    return 10**((T-T_ref)/z)

radius_fcalc = list()
radius_list = radius_index.tolist()
for i in range(len(radius_list)):
    if radius_list[i]%0.5 == 0:
        radius_fcalc.append(i)
        

Fcalc = list()
for i in range(len(Solutions)):
    integrand = Fcalc_integrand(Solutions[i])
    integral = np.empty((n_rows-1, len(radius_fcalc)))
    for i,col in enumerate(radius_fcalc):
        integral[:,i] = integrate.cumtrapz(integrand[:,col].flatten(), dx=delta_t)
    Fcalc.append(np.sum(integral, axis = 1))


### Figure 4

### set max time_max to 2400 sec

center = np.where(radius_index == 0)
point_sixfive = np.where(radius_index == radius-0.65)

T_80_center = Solutions[0][:, center]
T_80_sixfive = Solutions[0][:, point_sixfive]

x = time_index.flatten()/60
plt.figure(dpi = 2000)
plt.plot(x, T_80_center.flatten())
plt.plot(x, T_80_sixfive.flatten())
plt.xlim(0, 40)
plt.ylim(0, 100)
plt.xlabel("Time (min)")
plt.ylabel("Temperature (°C)")
plt.legend(["Center", "0.65 mm from surface"])
plt.savefig("Figure4_REDO.png", dpi = 2000)


### Figure 6
### set Tlist = [60, 80, 90], time_max = 112.5 sec

T_60_112s = np.asarray(Fcalc[0])/(60)
T_80_12s = np.asarray(Fcalc[1])/(60)
T_90_6s = np.asarray(Fcalc[2])/(60)


x = time_index[1:].flatten()
plt.figure(dpi = 2000)
plt.plot(x, T_60_112s.flatten())
plt.plot(x, T_80_12s.flatten())
plt.plot(x, T_90_6s.flatten())
plt.ylim(0, 32)
plt.xscale('log')
plt.xlim(1,300)
plt.xlabel("Time (s)")
plt.ylabel("Fcalc (min)")
plt.legend(["60 °C", "80 °C", "90 °C"])
plt.savefig("Figure6.png", dpi = 2000)

### Figure 7
### set Tlist = [60, 80, 90], time_max = 112.5 sec

sixty = np.where(time_index == 112.5)
eighty = np.where(time_index == 12.72)  
ninety = np.where(time_index == 6.355)


T_60_112s = Solutions[0][sixty]
T_80_12s = Solutions[1][eighty]
T_90_6s = Solutions[2][ninety]


x = radius_index.flatten()
plt.figure(dpi = 2000)
plt.plot(x, T_60_112s.flatten())
plt.plot(x, T_80_12s.flatten())
plt.plot(x, T_90_6s.flatten())
plt.xlim(0, 25)
plt.gca().invert_xaxis()
plt.ylim(0, 80)
plt.xlabel("Radius")
plt.ylabel("T (°C)")
plt.legend(["60 °C", "80 °C", "90 °C"])
plt.savefig("Figure7.png", dpi = 2000)