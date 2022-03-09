# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 08:59:33 2022

@author: zahra
"""

#Labs polynomial and spline interpolation
import matplotlib.pyplot as plt
import numpy as np 
from scipy import interpolate
from scipy.interpolate import interp1d 
from scipy.optimize import curve_fit

#Given a set of points (x_i, y_i)
y = [1,3,4,3,5,9,7,5,6,8,7,8,9,8,7,1,3,4,3,5,9,7,5,6,8,7,8,9,8,7]
n = len(y)
x = range(0, n)
xfit = np.arange(0, n-1, 0.001)
plt.plot(x , y, 'o')
plt.plot(x , y, label='Linear Spline Interpolant')

f = interpolate.CubicSpline(x, y, bc_type='natural')
y_= f(xfit) # evaluating the cubic spline for x_fit points
plt.plot(xfit , y_, label='Cubic-spline interpolant')

'''
#so, what does it help to know how to represent a polynomial in monomial basis? or as a matter of fact, any other basis
degree = n-1
A = np.vander(x,degree+1)

A = np.fliplr(A)
print(A)
#Ac = y, c = A-1y
coeffs = np.linalg.solve(A, y)
print(coeffs)

#polynomial evaluation
interp_m = 500
interp_x = np.linspace(0, n, interp_m)
interp_y = np.zeros(interp_m)
for ind, ix in enumerate(interp_x):
    interp_y[ind] = np.sum(coeffs * ix ** np.arange(0, degree+1))

plt.plot(interp_x, interp_y, label='Monomial basis interpolant order 4')
'''



#But then there is another function in numpy which also finds the curve. 
#So, whats the difference?
#least squares! 
poly = np.polyfit(x, y, deg = 10)
polyVal = np.polyval(poly, xfit)

plt.plot(xfit, polyVal, label='Least square interpolant')

plt.title("Spline and polynomial interpolations")

plt.legend()
















