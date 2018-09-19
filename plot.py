import matplotlib.pyplot as plt
import math

def plot(data, *labels, **params):
    '''create a line plot of the concentration of a chemical specie
    * data: pandas data frame, untidy format, time as the first column
    * *label: name of the column to plot
    * **params: parameters of the plot
    *   - log: [True, False]: logscaled y axis
    '''
    x = data["time"]
    if params.get("log", False):
        plot_function = plt.semilogy
    else:
        plot_function = plt.plot
    for label in labels:
        if label in data:
            y = data[label]
            plot_function(x, y, label=label)
            plt.xlabel("time (day)")
        else:
            print("WARNING: label {} not found in data".format(label))
    plt.ylabel("concentration (M)")
    plt.legend()
    plt.show()

def pHplot(data):
    x = data["time"]
    y = data["H+"].apply(lambda x: -math.log10(x))
    plt.plot(x, y, 2)
    plt.ylabel("pH")
    plt.show()
