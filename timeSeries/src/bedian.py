#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:31:09 2022

@author: apple
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d 

import pandas as pd
data = pd.read_csv("../data/data.csv") 
print(data.head())
zeroKm = data.loc[:,"0km"] 
print(zeroKm)
plt.plot(zeroKm/100., 'o',label='Water Level at 0 km ')


oneKm = data.loc[:,"1km"]
plt.plot(oneKm/100., 'o',label='Water Level at 1 km ')

'''
twoKm = data.loc[:,"2km"]
plt.plot(twoKm/100., 'o',label='Water Level at 2 km ')
'''

threeKm = data.loc[:,"3km"]
#removing anomalies from data 3km
threeKm = threeKm.loc[threeKm != 300]
plt.plot(threeKm/100., 'o',label='Water Level at 3 km ')
plt.title("Water level recorded from sensors")

'''
mblKm = data.loc[:,"MBL"]
plt.plot(mblKm/100., 'o',label='Water Level at midpoint (1.5) km ')
'''

plt.xlabel ( 'Time (minutes)' )
plt.ylabel ( 'Water level [h]' )
plt.legend()
plt.savefig("./results/data.png")

'''
______________________________________________________________________________

                            Interpolation done below.
            Variable deg set control for the degree of polynomial

______________________________________________________________________________
'''


# Interpolating a polynomial at 3 km

t3 = threeKm.index

poly3 = np.polyfit(t3, threeKm / 100., deg = 5)
polyVal3 = np.polyval(poly3, t3)

print(poly3)
plt.clf()
plt.plot(threeKm/100., 'o',label='Water Level at 3 km ')
plt.plot(t3, polyVal3, color = 'red', label='Interpolant of degree 5')
plt.xlabel ( 'Time (minutes)' )
plt.ylabel ( 'Water level [h]' )
plt.legend()
plt.savefig("./results/data_3km.png")
plt.show()

#TODO: Fit a third or fourth order polynomial, find the polynomial coeffiecients.  


# Interpolating a polynomial at 0 km

t = zeroKm.index

poly = np.polyfit(t, zeroKm / 100., deg = 5)
polyVal = np.polyval(poly, t)

plt.clf()
plt.plot(zeroKm/100., 'o',label='Water Level at 0 km ')
plt.plot(t, polyVal, color = 'red', label='Interpolant of degree 5')
plt.xlabel ( 'Time (minutes)' )
plt.ylabel ( 'Water level [h]' )
plt.legend()
plt.savefig("./results/data_0km.png")
plt.show()
