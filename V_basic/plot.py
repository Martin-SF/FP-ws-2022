# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.stats import sem
# import scipy.constants as const
# from scipy.constants import physical_constants #von arne und max
import numpy as np
# import sympy as sy
# from uncertainties import ufloat
# from uncertainties import correlated_values
# from uncertainties.umath import *
# import uncertainties.unumpy as unp
import sys
sys.path.append("../V_basic/content/fix/")
from FP_plot_helpers import *  # alle unsere plot.py funktionen


# Datentyp mit Fehler: ufloat(,)
# Datenarray mit Fehler: unp.uarray(,)
# Array mit Fehlern: unp.uarray(['werte'],['fehler'])
# Mittelwert berechnet sich mit: np.mean
# Fehler des Mittelwerts berechnet sich mit np.std
# uarray: ist ein Fehlerarray, Zugriff mit nominal_values und std_devs

# Beispiele
# U, N, I = np.genfromtxt('data/a.txt', unpack=True)
# data = array([U, N, I]).T
# print(make_LaTeX_table(data, [r'$U \;/\; \si{\volt}$',
# 						r'$N$', r'$I \;/\; \si{\ampere}$']))