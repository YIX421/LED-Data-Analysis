# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:11:11 2024

@author: Yiman Xu

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import integrate as intg
import pandas as pd
import math


df = pd.read_csv(r'your file path of LED EL spectrum')
result = pd.read_csv(r'your file path of LED V-I-L data')


# Gaussian
# Define the gaussian function
def gauss(x,amp,mu,sigma,offset):
    return amp*np.exp(-(x-mu)**2/(2*sigma**2))+offset

x = df['wave']
y0 = df['lumi']
# Normarlize data
y = (y0 - y0.min())/(y0.max()-y0.min())
                     
# Plot the normalized electroluminescence
plt.plot(x,y,color='red')

plt.title('Electroluminescence spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized electroluminescence (a.u.)')

plt.show()

# Estimate initial values p0
n = len(x)
mean = sum(x*y)/sum(y)
sigma = np.sqrt(abs(sum((x-mean)**2*y)/sum(y)))

p0 = [-max(y),mean,sigma,min(x)+((max(x)-min(x)))/2]

coeff, var_matrix = curve_fit(gauss, df['wave'],y,p0=p0) 
print(coeff)
print(var_matrix)

# Get the fitted curve
hist_fit = gauss(df['wave'], *coeff)

# Plot it if you want to see the fitting quality
plt.plot(df['wave'],y,color='red')
plt.plot(df['wave'], hist_fit,color='blue')

plt.title('Electroluminescence fitting')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized electroluminescence (a.u.)')

plt.show()

# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
print('Height = ', coeff[0])
print('Fitted mean = ', coeff[1])
print('Fitted standard deviation = ', coeff[2])
print('Fitted offset =', coeff[3])


# Get c (the weight average)
Slamda = lambda a: coeff[0]*np.exp(-(a-coeff[1])**2/(2*coeff[2]**2)) + coeff[3]
Int_Qlamda=intg.quad(lambda a:-1.47818 + 0.00538*a - 2.99871*10**(-6)*a**2,350,800)

Int_Slamda=intg.quad(Slamda,350,800)
TotalInt= intg.quad(lambda a:(-1.47818 + 0.00538*a - 2.99871*10**(-6)*a**2) * (coeff[0]*np.exp(-(a-coeff[1])**2/(2*coeff[2]**2)) + coeff[3]),350,800)

c=TotalInt[0]/Int_Slamda[0]

print('C =',c)

# D mm is the distance from LED to detector
D = 100

# A mm^2 is the active area of LED
A = 4

# Transimpedance gain
TG = 4.75*10**5

# H e s^-1 V^-1 is the sensitivity of the gain transimpedance amlifier
H = 1/(TG*1.6*10**(-19))

# Aphd is the active area of the photodetector
Aphd = 75.4

# Get Ωphd sr
Omega = 2 * math.pi * (1-np.cos((Aphd/math.pi)**0.5/D))

Plamda = lambda a:698.83958952*np.exp(-(a-560.18504094)**2/(2*43.80189435**2)) - 1.09653258
PlamdaSlamda = lambda a:(698.83958952*np.exp(-(a-560.18504094)**2/(2*43.80189435**2)) - 1.09653258)*(coeff[0]*np.exp(-(a-coeff[1])**2/(2*coeff[2]**2)) + coeff[3])*(6.626e-34*2.998e+08*10**9/a)

TotalInt1 = intg.quad(PlamdaSlamda,350,800)
k = TotalInt1[0]/Int_Slamda[0]

print('K =',k)

plt.show()

# Get the results
Vled = result['V']
Iled = result['I']
Vphd = result['L']

# Get current density J mA cm^-2
J = (Iled * 1000) / (A * 0.01)

result['J mA cm^-2'] = J

# Get photon flux Φphd photons s^-1 sr^-1
phi = (H * Vphd)/(c * Omega)

result['photon flux s^-1 sr^-1'] = phi

# Get EQE %
EQE = (1.602*10**(-19) * math.pi * phi * 100)/Iled

result['EQE %'] = EQE

# Get luminous density L cd m^-2
L = (H * k * Vphd)/(c * Omega * A*10**(-6))

result['luminous density L cd m^-2'] = L

# Write all results to the csv file
result.to_csv(r'enter the your file path of LED V-I-L data again', index=False)


# Plot J-V-L
fig, ax1 = plt.subplots()

ax1.set_xlabel('Voltage (V)')
ax1.set_ylabel('Current density (mA cm⁻²)', color='black')
ax1.plot(Vled, J, color='black')
ax1.tick_params(axis='y', labelcolor='black')
plt.yscale('log',base=10)
plt.ylim(0.0001,1000)
plt.xlim(0,8)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Luminance (cd m⁻²)', color='red')  # we already handled the x-label with ax1
ax2.plot(Vled, L, color='red')
ax2.tick_params(axis='y', color='red', labelcolor='red')
plt.yscale('log',base=10)
plt.ylim(1,100000)
 
fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.title('J-V-L')

plt.show()

# Plot EQE-J
plt.plot(L, EQE,color='darkorange')
plt.title('EQE-L')
plt.xlabel('Luminance (cd m⁻²)')
plt.ylabel('EQE(%)')
plt.ylim(0,20)
plt.xlim(0,8000)

plt.show()

# I-V
plt.plot(Vled, Iled*10**4,color='black')
plt.title('I-V')
plt.xlabel('Voltage (V)')
plt.ylabel('Current ($10^4$ A)')

plt.show()

# L-V
plt.plot(Vled, L, color='black')
plt.title('L-V')
plt.xlabel('Voltage (V)')
plt.ylabel('Luminance (cd m⁻²)')

plt.show()