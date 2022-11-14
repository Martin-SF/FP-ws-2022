import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import correlated_values
import uncertainties.unumpy as unp

# ersetzt die matplotlibrc
# verweis auf header-matplotlib.tex, benötigt:
# TEXINPUTS="$(call translate,$(pwd):)" python plot.py
# (in makefile integriert)
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams.update({
 'font.family': 'serif',
 'text.usetex': True,
 'pgf.rcfonts': False,
 'pgf.texsystem': 'lualatex',

 'figure.figsize': '5.78, 3.57',
 'font.size': '11',
 'legend.fontsize': 'medium',
 'xtick.labelsize': '9',
 'ytick.labelsize': '9',

 'pgf.preamble': r'\input{content/fix/header-matplotlib.tex}',
})


def f(x, a, b):  # andere fit funktion
    return a * x + b


def sigmoid(x, a, b, c):
    y = a / (1 + np.exp(-(x-b))) + c
    return y


def f_2(x, a, b, c):
    return a * np.exp(-b * x) + c


def f_3(x, a, b, c):
    return a * np.sqrt(abs(b*x)) + c


def gauss(x, a, b, c, x_0):
    return a + b * np.exp(- ((x-x_0)/c ** 2))


def potenz(x, a, b, c, d):
    return a * (x-b)**c + d


def fit_plt(s, x, x_s, y, y_s, errY='default', lin_length='default'):

    # Fitting ####################################
    my_func = f
    params, covariance_matrix = curve_fit(my_func, x, y)  # p0 eventl. setzen
    params = correlated_values(params, covariance_matrix)

    # print results ####################################
    print('\nPlot: %s' % s)
    from inspect import signature
    sig_params = signature(f).parameters
    for i in range(0, len(sig_params)-1):
        print('Parameter %s = %s' % (list(sig_params.keys())[i+1], str(params[i])))
        # iteriert über die Argumente der verwendeten funktion
    
    # plot results ####################################
    if (lin_length == 'default'):
        lin_length = x[-1]
    x_plot = np.linspace(0, lin_length)

    plt.plot(x, y, 'k+')
    plt.plot(x_plot, my_func(x_plot, *unp.nominal_values(params)),
             'r-', label='lin. Fit')
    if (errY != 'default'):
        plt.errorbar(x, y, yerr=errY, fmt="none", capsize=3, capthick=1, ms=9,
                 markerfacecolor="black", label='Fehlerbalken')
    plt.xlabel(x_s)
    plt.ylabel(y_s)

    # plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    # print('Plot saving to %s ...' % ('pic/'+s+'.pdf'))
    plt.savefig('pic/'+s+'.pdf')
    print('Plot successfully saved to %s' % ('pic/'+s+'.pdf'))
    plt.close('all')


def plotter(s, x, x_s, y, y_s, a, b):
    plt.plot(x, y, 'b', linewidth=1, label='Messdaten')
    plt.xlabel(x_s)
    plt.ylabel(y_s)
    plt.xlim([a, b])
    # plt.ylim([0,500])
    # plt.grid() # Gitter
    # plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(''+s+'.pdf')
    plt.close('all')


# array und int32 warsch aus numpy
def make_LaTeX_table(data, header, flip='false', onedim='false'):
    output = '\\begin{table}\n\\centering\n\\begin{tabular}{'
    # Get dimensions
    if(onedim == 'true'):
        if(flip == 'false'):

            data = array([[i] for i in data])

        else:
            data = array([data])

    row_cnt, col_cnt = data.shape
    header_cnt = len(header)

    if(header_cnt == col_cnt and flip == 'false'):
        # Make Format

        for i in range(col_cnt):
            output += 'S'
        output += '}\n\\toprule\n{' + header[0]
        for i in range(1, col_cnt):
            output += '} &{ ' + header[i]
        output += ' }\\\\\n\\midrule\n'
        for i in data:
            if(isinstance(i[0], (int, float, int32))):
                output += str(i[0])
            else:
                output += ' ${:L}$ '.format(i[0])
            for j in range(1, col_cnt):
                if(isinstance(i[j], (int, float, int32))):
                    output += ' & ' + str(i[j])
                else:
                    output += ' & ' + str(i[j]).replace('/', '')

            output += '\\\\\n'
        output += '\\bottomrule\n\\end{tabular}\n\\caption{}\n\\label{}\n\\end{table}\n'

        return output

    else:
        return 'ERROR'
