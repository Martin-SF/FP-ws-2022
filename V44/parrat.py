# %%
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

a, R_G = np.genfromtxt('build/R_G.csv',delimiter=',',unpack=True)

# Ideale Fresnelreflektivität
R_ideal = (0.223 / (2 * a))**4

# Betrag des Wellenvektors
k = 2*np.pi/1.54e-10  

# manuelles Raten der Parameter
delta = [0, 0, 0, 0]
delta[2] = 1.0e-6
delta[3] = 7.4e-6

sigma = [0, 0, 0]
sigma[1] = 6.55e-10
sigma[2] = 8.15e-10

beta = [0, 0, 0, 0]
beta[2] = 0.05e-7
beta[3] = 0.6e-6
beta[2] = 0
beta[3] = 0

n = [0, 0, 0, 0]
n[1] = 1
n[2] = 1. - delta[2] + 1j * beta[2]
n[3] = 1. - delta[3] + 1j * beta[3]

z = [0, 0, 0]
z[2] = 8.1e-8

a_i = np.deg2rad(a)

def X(j):
    if (j==3):
        return 0
    else:
        return (
            np.exp(-2 * 1j * kz(j) * z[j]) *
            (r(j) + 
            X(j+1) * np.exp(2 * 1j * kz(j+1) * z[j])) /
            (1 + r(j) * 
            X(j+1) * np.exp(2 * 1j * kz(j+1) * z[j]))
        )

def kz(j):
    return k * np.sqrt(n[j]**2 - np.cos(a_i)**2)

def r(j):
    return (
        r_normal(j) * np.exp(-2 * kz(j) * kz(j+1) * sigma[j]**2)
    )

def r_normal(j):
    return (
        (kz(j) - kz(j+1)) / (kz(j) + kz(j+1))
    )

R_parr = np.abs(X(1))**2

peaks_mask = (a>=0.3) & (a<=1.19)
def f(x,b,c):
    return b*x+c

params, pcov = curve_fit(f,a[peaks_mask],np.log(R_G[peaks_mask]))
R_fit = np.exp(f(a[peaks_mask],*params))

# Minima der Kissig-Oszillation
i_peaks, peak_props = find_peaks(-(R_G[peaks_mask]-R_fit), distance=7)
i_peaks += np.where(peaks_mask)[0][0]

# Fresnelreflektivität Manueller Fit Reflektivitätsscan - Diffuser Scan)G Oszillationsminima
plt.plot(a, R_ideal, '-',color='pink', label='Fresnelreflektivität von Si')
plt.plot(a, R_parr, '-', label='Manueller Fit')
plt.plot(a, R_G, '-', label=r'(Reflektivitätsscan - Diffuser Scan)$/G$')
plt.plot(a[i_peaks], R_G[i_peaks], 'kx', label='Oszillationsminima',alpha=0.8)
plt.xlabel(r'$\alpha_\text{i} \:/\: \si{\degree}$')
plt.ylabel(r'$R$')
plt.yscale('log')
plt.legend(loc='upper right',prop={'size': 8})
plt.tight_layout()
plt.savefig('build/plot_messung2_parrat.pdf')
plt.clf()