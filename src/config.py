import matplotlib.pyplot as plt
from cycler import cycler

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

def setup_matplotlib():
    plt.rcParams['figure.figsize']=[32, 18]
    plt.rcParams['font.size'] = 16
    plt.rcParams['font.weight'] = 'bold'

    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.prop_cycle'] = cycler(color=CB_color_cycle)

    plt.rcParams['lines.linewidth'] = 3.0

    plt.style.use('seaborn-whitegrid')