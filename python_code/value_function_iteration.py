import numpy as np
import numpy.matlib as mt
import pandas as pd
from highcharts import Highchart
H = Highchart(width=750, height=600)
H1 = Highchart(width=750, height=600)

A = 1.0
alpha = 0.36
beta = 0.9

kss = (alpha*beta)**(1/(1-alpha))   # steady state

# comment for Part (d)
L = 0.95                           # factor lower bound
U = 1.05                           # factor upper bound
n = 2                              # number of grid points

# uncomment for Part (d)
# L = 0.6                         # idem
# U = 1.4                         # idem
# n = 100                         # idem

k = np.linspace(L * kss, U * kss, n)

criteria = 1.0                     # init criterion for stopping iterations
tol = 1e-6                         # tolerance
it = 0                             # number of iterations


''' initialization of endogeneous variables '''

v = np.zeros((n, 1))                     # v(i): value function at k(i)
tv = np.zeros((n, 1))                    # tv(i): mapped value function at k(i)
k_prime = np.zeros((n, 1))               # k_prime(i): next-period capital given current state k(i)


c = A * mt.repmat(np.power(k, alpha), n, 1).transpose() - mt.repmat(k, n, 1)  # cons(i,j): consumption given state k(i) and decision k(j)

c[c < 0] = 0.0                                                                # gives -Inf utility
util = np.log(c)                                                              # util(i,j): current utility at state k(i) and decision k(j)


while criteria > tol:
    v_vec = mt.repmat(v, 1, n).transpose()
    util_v_vec = util + beta * v_vec
    tv = np.array([util_v_vec.max(1)]).T
    k_prime = k[util_v_vec.argmax(1)]
    criteria = np.max(np.abs(tv - v))                # criteria for tolerance
    v = tv                                           # update value function
    it = it+1                                        # update number of iterations


''' Part B. parameters analytical solution '''

E = 1/(1-beta)*(np.log(A*(1-alpha*beta))+alpha*beta/(1-alpha*beta)*np.log(A*alpha*beta))

F = alpha/(1-alpha*beta)


''' Part C. Graphing using HighCharts. '''
# Graph for E+F*np.log(k)

df_discrete = pd.DataFrame({'k': k, 'discrete': E+F*np.log(k)})
df_discrete = df_discrete[['k', 'discrete']]
discrete = df_discrete.values.tolist()

df_v = pd.DataFrame({'v': v.flatten(), 'k': k})
v = df_v.values.tolist()


options = {
    'title': {
        'text': 'Value function iteration'
    },
    'xAxis': {
        'title': {
            'text': "K"
        }
    },
    'yAxis': {
        'title': {
            'text': "Value of Capital"
        }
    },
    'tooltip': {
        'crosshairs': True,
        'shared': True,
    },
    'legend': {
    }
}

H.set_dict_options(options)


H.add_data_set(discrete, 'scatter', 'Discrete', color='rgba(223, 83, 83, .5)')
H.add_data_set(v, 'line', 'Continuous', zIndex=1, marker={
                'fillColor': 'white',
                'lineWidth': 2,
                'lineColor': 'Highcharts.getOptions().colors[0]'
            })

html_str = H.htmlcontent.encode('utf-8')

html_file = open("chart.html", "w")
html_file.write(html_str)
html_file.close()


# Graph for alpha*beta*A*k.^(alpha)

df_discrete = pd.DataFrame({'k': k, 'discrete': alpha*beta * A * np.power(k, alpha)})
df_discrete = df_discrete[['k', 'discrete']]
discrete = df_discrete.values.tolist()

df_k_prime = pd.DataFrame({'k_prime': k_prime, 'k': k})
k_prime = df_k_prime.values.tolist()

options = {
    'title': {
        'text': 'Value function iteration'
    },
    'xAxis': {
        'title': {
            'text': "K"
        }
    },
    'yAxis': {
        'title': {
            'text': "Next Period Capital"
        }
    },
    'tooltip': {
        'crosshairs': True,
        'shared': True,
    },
    'legend': {
    }
}

H1.set_dict_options(options)


H1.add_data_set(discrete, 'scatter', 'Discrete', color='rgba(223, 83, 83, .5)')
H1.add_data_set(k_prime, 'line', 'Continuous', zIndex=1, marker={
                'fillColor': 'white',
                'lineWidth': 2,
                'lineColor': 'Highcharts.getOptions().colors[0]'
            })

html_str = H1.htmlcontent.encode('utf-8')

html_file = open("chart2.html", "w")
html_file.write(html_str)
html_file.close()
