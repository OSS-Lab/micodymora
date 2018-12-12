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
            plt.xlabel("time (hour)")
        else:
            print("WARNING: label {} not found in data".format(label))
    plt.ylabel("concentration (M)")
    plt.legend()
    plt.show()

def fraction_plot(data, *labels, **params):
    x = data["time"]
    y = data[[*labels]]
    y_frac = y.divide(y.sum(axis=1), axis=0)
    plt.stackplot(x, *[y_frac[label] for label in labels], labels=labels)
    plt.xlabel("time (hour)")
    plt.legend()
    plt.show()

