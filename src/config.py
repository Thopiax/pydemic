import matplotlib.pyplot as plt

def setup_matplotlib():
    plt.rcParams['figure.figsize']=[32, 18]
    plt.rcParams['font.size']=16
    plt.rcParams['font.weight']='bold'
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.labelsize'] = 16

    plt.style.use('seaborn-whitegrid')