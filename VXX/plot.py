import matplotlib.pyplot as plt
import numpy as np

# from scipy.optimize import curve_fit
# from scipy.stats import sem
# import scipy.constants as const
# from scipy.constants import physical_constants #von arne und max

# import sympy as sy

# from uncertainties import ufloat
# from uncertainties import correlated_values
# from uncertainties.umath import *
# import uncertainties.unumpy as unp
# import sys
# sys.path.append("../V_basic/content/fix/")
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

# beispiele aus toolbox 2022

# x = np.linspace(0, 10, 1000)
# y = x ** np.sin(x)

# plt.subplot(1, 2, 1)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')

# plt.subplot(1, 2, 2)
# plt.plot(x, y, label='Kurve')
# plt.xlabel(r'$\alpha \mathbin{/} \unit{\ohm}$')
# plt.ylabel(r'$y \mathbin{/} \unit{\micro\joule}$')
# plt.legend(loc='best')

# # in matplotlibrc leider (noch) nicht möglich
# plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
# plt.savefig('build/plot.pdf')