# %%
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
    'pgf.preamble': r'\usepackage{siunitx}',
})
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from scipy.optimize import curve_fit, root
from uncertainties import unumpy
import scipy.constants.constants as const
from uncertainties import ufloat
from matplotlib.ticker import FormatStrFormatter
from scipy.signal import find_peaks
from scipy.stats import sem


#############################
#Auswertung des Detektorscans
#############################

a_det, I_det = np.genfromtxt('data/gaussdetektorscan0.UXD',skip_header= 57, unpack=True)


# Gaußfunktion als Ausgleichskurve
def gauss(x,a,b,sigma,mu):
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2)) + b


print('Detectorscan')
p0 = [10**6, 0, 10**(-2), 10**(-2)]
params, covar =  curve_fit(gauss,a_det, I_det, p0=p0)           
uparams = unumpy.uarray(params, np.sqrt(np.diag(covar)))
for i in range(0, len(uparams)):
    print(chr(ord('a') + i), "=" , uparams[i])
print()
a_lin = np.linspace(np.min(a_det), np.max(a_det),100000)
intensity_gauss = gauss(a_lin, *params)

# Maximale Intensität bestimmen 
I_max = np.max(intensity_gauss)
print('Maximale Intensität: ' , I_max)
# Halbwertsbreite bestimmen
left_FWHM = root(lambda x: gauss(x,*params)-(I_max/2), x0=-0.01).x[0]
right_FWHM = root(lambda x: gauss(x,*params)-(I_max/2), x0=0.1).x[0]
FWHM = np.absolute(right_FWHM - left_FWHM)
print('Halbwertsbreite: ', FWHM)

#Plotten
plt.plot(a_det, I_det,'rx' ,label='Messdaten')
plt.plot(a_lin,intensity_gauss,label='Gaußverteilung')
plt.plot([left_FWHM, right_FWHM], [I_max/2, I_max/2], 'b--', label='Halbwertsbreite')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.xlabel(r'$\alpha / \si{\degree}$')
plt.legend(loc='best')
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('build/detectorscan.pdf')
# plt.show()
plt.clf()

# %%
############
## Z-Scan 1 (G faktor)
############
print('Z_Scan 1')

z, I_z = np.genfromtxt('data/zscan0.UXD',skip_header= 57, unpack=True)

# Strahlbreite Ablesen
i_d = [10,15]
d0 = np.abs(z[i_d[0]]-z[i_d[1]]) # mm
print('d_0: ', d0)
# Plotten
plt.axvline(z[i_d[0]],color='blue',linestyle='dashed',label='Strahlgrenzen')
plt.axvline(z[i_d[1]],color='blue',linestyle='dashed')
plt.plot(z, I_z, 'rx', label='Messdaten')
plt.xlabel(r'$z \:/\: \si{\milli\meter}$')
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.legend(loc='best')
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('build/zscan.pdf')
plt.clf()


# %%
##################
## Rocking-Scan 1
##################
print('Rocking Scan')
a_roc, I_roc = np.genfromtxt('data/rocking0.UXD',skip_header= 57, unpack=True)

# Geometriewinkel ablesen
i_g = [32,-33]
a_g = np.mean(np.abs(a_roc[i_g]))
print('alpha_g_l: ',a_roc[i_g[0]])
print('alpha_g_r: ',a_roc[i_g[1]])
print('alpha_g: ', a_g)

D = 20 #mm

a_g_berechnet = np.rad2deg(np.arcsin(d0/D))
print('alpha_theorie: ' , a_g_berechnet)

# #Plotten
plt.axvline(a_roc[i_g[0]],color='blue',linestyle='dashed',label='Geometriewinkel')
plt.axvline(a_roc[i_g[1]],color='blue',linestyle='dashed')
plt.plot(a_roc, I_roc, 'rx', label='Messdaten')
#plt.plot(a_roc2, I_roc2, 'bo', label='Messdaten y-alligned')
plt.xlabel(r'$\alpha \:/\: \si{\degree}$')
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.legend()
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('build/rockingscan.pdf')
#plt.show()
plt.clf()

# %%
##########################################################
#Auswertung des Reflektivitätsscans und des diffusen Scans
##########################################################

# Reflektivitätsscan:
a_refl, I_refl = np.genfromtxt('data/messung1.UXD',skip_header= 57, unpack=True)
# Diffuser Scan
a_diff, I_diff = np.genfromtxt('data/messung2.UXD',skip_header= 57, unpack=True)

#Winkel sind die gleichen 
a = a_refl

# Anfang und Ende abschneiden
a_min = 0.01 
# a_max = 1.10
a_max = 1.10
mask = (a >= a_min) & (a <= a_max)
a = a[mask]
I_refl = I_refl[mask]
I_diff = I_diff[mask]
# Eingehende Intensität als das Maximum vom Detektorscan
# aber mit 5 multipliziert weil nun statt 1s 5s pro Winkel gemessen wurden

# Reflektivität: R=I_r/I_max
R_refl =  I_refl / (I_max * 5)
R_diff =  I_diff / (I_max * 5)

# diffusen Scan abziehen
R = R_refl - R_diff

# Geometriefaktor
G = np.ones_like(R)
G[a < a_g] = D/d0 * np.sin( np.deg2rad(a[a < a_g]) )

# um Geometriefaktor korrigieren
R_G = R / G
np.savetxt('build/R_G.csv', list(zip(a, R_G)), header='a_i,R_G', fmt='%.4f,%.10e')

# Ideale Fresnelreflektivität
a_c_Si = 0.223
R_ideal = (a_c_Si / (2 * a))**4

## Peaks finden
# Curve Fit für find_peaks
peaks_mask = (a>=0.3) & (a<=1.19)
def f(x,b,c):
    return b*x+c

params, pcov = curve_fit(f,a[peaks_mask],np.log(R_G[peaks_mask]))
R_fit = np.exp(f(a[peaks_mask],*params))

# Minima der Kissig-Oszillation finden
i_peaks, peak_props = find_peaks(-(R_G[peaks_mask]-R_fit), distance=7)
i_peaks += np.where(peaks_mask)[0][0]

# Schichtdicke bestimmen
lambda_ = 1.54*10**(-10) # m

delta_a = np.diff(np.deg2rad(a[i_peaks]))
delta_a_mean = ufloat(np.mean(delta_a),sem(delta_a))
print('delta_a_mean: ', delta_a_mean)
d = lambda_ / (2*delta_a_mean)
print('Schichtdicke: ', d)


# %%
####################
# Parrat Algorithmus
####################

# a_i : Einfallswinkel
# n_i: Brechundsindizes
# n_1 : Luft; n_2 : Schicht , n_3 : Substrat
# sigma_i : Rauigkeiten; _1: Schicht; _2 : Substrat
# z1 : 0; z_2 : Schichtdicker
# k= 2 pi / lambda : Betrag des Wellenvektors

#Konstanten
n1 = 1.
z1 = 0.
k = 2*np.pi/lambda_ 
# Schätzwerte für die Parameter 
delta2 = 1.0*10**(-6)#0.80*10**(-6)# 0.45*10**(-6)#Gering -> PS schicht kaum noch vorhanden
delta3 = 7.4*10**(-6)#11.05*10**(-6)#7.6*10**(-6)#Literaturwert
sigma1 = 6.55*10**(-10)#6.05*10**(-10)#2.3*10**(-10) # m ; Rauigkeit müssen abgeschätzt werden
sigma2 = 8.15*10**(-10)#5.90*10**(-10)#0.4*10**(-10) # m ; 
z2 =  8.1*10**(-8)#8.90*10**(-8)#8.7*10**(-8) # m 8.70
# beta2 = 0.05*10**(-7)
# beta3 = 0.6*10**(-6)
beta2 = 0
beta3 = 0

def parrat(a_i,delta2,delta3,sigma1,sigma2,z2, beta2,beta3):
    n2 = 1. - delta2 + 1j * beta2
    n3 = 1. - delta3 + 1j * beta3

    a_i = np.deg2rad(a_i)

    #kz1 = k * np.sqrt(np.abs(n1**2 - np.cos(a_i)**2))
    #kz2 = k * np.sqrt(np.abs(n2**2 - np.cos(a_i)**2))
    #kz3 = k * np.sqrt(np.abs(n3**2 - np.cos(a_i)**2))
    kz1 = k * np.sqrt(n1**2 - np.cos(a_i)**2)
    kz2 = k * np.sqrt(n2**2 - np.cos(a_i)**2)
    kz3 = k * np.sqrt(n3**2 - np.cos(a_i)**2)
    
    r12 = (kz1 - kz2) / (kz1 + kz2) * np.exp(-2 * kz1 * kz2 * sigma1**2)
    r23 = (kz2 - kz3) / (kz2 + kz3) * np.exp(-2 * kz2 * kz3 * sigma2**2)

    x2 = np.exp(-2j * kz2 * z2) * r23
    x1 = (r12 + x2) / (1 + r12 * x2)
    R_parr = np.abs(x1)**2

    return R_parr

params = [delta2,delta3,sigma1,sigma2,z2,beta2,beta3]
R_parr = parrat(a, *params)


# Kritischer Winkel
a_c2 = np.rad2deg(np.sqrt(2*delta2))
a_c3 = np.rad2deg(np.sqrt(2*delta3))
print('a_c2:', a_c2)
print('a_c3:', a_c3)

#R_ideal[a <= a_c3] = np.nan
#R_parr[a <= a_c3] = np.nan

# %%
############
## Plotten
############
# Reflektivitäts Scan Plotten
print('Plot: Mess-Scan...')
#mpl.rcParams['lines.linewidth'] = 0.9
#mpl.rcParams['axes.grid.which'] = 'major'
# plt.axvline(a_c2, linewidth=0.6, linestyle='dashed', color='blue', label=r'$\alpha_\text{c,PS},\alpha_\text{c,Si}$')
# plt.axvline(a_c3, linewidth=0.6, linestyle='dashed', color='blue')
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

plt.plot(a, R_refl, '-', label='Reflektivitätsscan ')
plt.plot(a, R_diff, '-', label='Diffuser Scan ')
plt.plot(a, R, '-', label='Reflektivitätsscan - Diffuser Scan ')
plt.xlabel(r'$\alpha_\text{i} \:/\: \si{\degree}$')
plt.ylabel(r'$R$')
plt.yscale('log')
plt.legend(loc='upper right',prop={'size': 8})
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('build/plot_messung1_diff.pdf')
plt.clf()

##########
# %%
