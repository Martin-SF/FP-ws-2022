#ersetzt die matplotlibrc
# verweis auf header-matplotlib.tex, benötigt:
#	TEXINPUTS="$(call translate,$(pwd):)" python plot.py 
#	(in makefile integriert)
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
'font.family': 'serif',
'text.usetex': True,
'pgf.rcfonts': False,
'pgf.texsystem': 'lualatex',

'figure.figsize' : '5.78, 3.57',
'font.size' : '11',
'legend.fontsize' : 'medium',
'xtick.labelsize' : '9',
'ytick.labelsize' : '9',

'pgf.preamble': r'\input{content/fix/header-matplotlib.tex}',
})

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import sem
import scipy.constants as const
#from scipy.constants import physical_constants #von arne und max

import numpy as np

import sympy as sy

from uncertainties import ufloat, correlated_values
from uncertainties.umath import *
import uncertainties.unumpy as unp


#Datentyp mit Fehler: ufloat(,)
#Datenarray mit Fehler: unp.uarray(,)
# Array mit Fehlern: unp.uarray(['werte'],['fehler'])
#Mittelwert berechnet sich mit: np.mean
#Fehler des Mittelwerts berechnet sich mit np.std
#uarray: ist ein Fehlerarray, Zugriff mit nominal_values und std_devs

# Beispiele
# U, N, I = np.genfromtxt('data/a.txt', unpack=True)
# data = array([U,N,I]).T
# print(make_LaTeX_table(data,[r'$U \;/\; \si{\volt}$', r'$N$', r'$I \;/\; \si{\ampere}$']))

def fit_plt(s,x,x_s,y,errY,y_s,lin_length):

	def f(x, a, b): #andere fit funktion
		return a * x + b
	def sigmoid(x, a, b, c):
		y = a / (1 + np.exp(-(x-b))) + c
		return y
	def f_2(x, a, b, c):
		return a * np.exp(-b * x) + c
	def f_3(x, a, b, c):
		return a * np.sqrt(abs(b*x)) + c
	def gauss(x, a, b, c, x_0):
		return a + b * np.exp(- ( (x-x_0)/c )**2 )
	def potenz(x, a, b, c, d):
		return a * (x-b)**c +d


	used_function = f
    #ZUM WECHSELN DER FUNKTION used funtion ändern und stratwerte bei curve_fit

	params, covariance_matrix = curve_fit(used_function, x, y, p0=(-1,0,2, 1000))
	                                 #curve_fit(used_function, x, y, p0=(-1,1,0))
                                 #curve_fit(used_function, x, y)
                                 #curve_fit(used_function, x, y, p0=(70,1000,1, 300)) //Test für Gauss-Fit

	params = correlated_values(params, covariance_matrix) #direkt in unp.uarray
	print('Methode : ' + s)
	print('a =', params[0])
	print('b =', params[1])
	print('c =', params[2])
	print('d =', params[3])
	#plt.rcParams['figure.figsize'] = (10, 8)
	#plt.rcParams['font.size'] = 16

	x_plot = np.linspace(0, lin_length) #max von x werten = lin_length

	plt.plot(x, y, 'k+')
	plt.errorbar(x, y, yerr=errY, fmt="none", capsize=3, capthick=1, ms=9, markerfacecolor="black", label='Fehlerbalken')


	plt.plot(x_plot, used_function(x_plot, *params), 'r-', label='lin. Fit', linewidth=1) #dicke
	#plt.plot(x, used_function(x, *params), 'r-', label='lin. Fit', linewidth=1)
	plt.xlabel(x_s)
	plt.ylabel(y_s)
	plt.xlim([0, lin_length])
	#plt.title(s)
	#plt.grid() # Gitter

	plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig('pic/'+s+'.pdf')
	plt.close('all')


def plotter(s,x,x_s,y,y_s,a,b):
	plt.plot(x, y, 'b', linewidth=1, label='Messdaten')
	plt.xlabel(x_s)
	plt.ylabel(y_s)
	plt.xlim([a, b])
	#plt.ylim([0,500])
	#plt.grid() # Gitter
	#plt.legend(loc="best")
	plt.tight_layout()
	plt.savefig(''+s+'.pdf')
	plt.close('all')


def make_LaTeX_table(data,header, flip= 'false', onedim = 'false'):
    output = '\\begin{table}\n\\centering\n\\begin{tabular}{'
    #Get dimensions
    if(onedim == 'true'):
        if(flip == 'false'):

            data = array([[i] for i in data])

        else:
            data = array([data])

    row_cnt, col_cnt = data.shape
    header_cnt = len(header)

    if(header_cnt == col_cnt and flip== 'false'):
        #Make Format

        for i in range(col_cnt):
            output += 'S'
        output += '}\n\\toprule\n{'+ header[0]
        for i in range (1,col_cnt):
            output += '} &{ ' + header[i]
        output += ' }\\\\\n\\midrule\n'
        for i in data:
            if(isinstance(i[0],(int,float,int32))):
                output += str( i[0] )
            else:
                output += ' ${:L}$ '.format(i[0])
            for j in range(1,col_cnt):
                if(isinstance(i[j],(int,float,int32))):
                    output += ' & ' + str( i[j])
                else:
                    output += ' & ' + str( i[j]).replace('/','')

            output += '\\\\\n'
        output += '\\bottomrule\n\\end{tabular}\n\\caption{}\n\\label{}\n\\end{table}\n'

        return output

    else:
        return 'ERROR'
