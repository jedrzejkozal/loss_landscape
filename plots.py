import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_levels(x, y, z):
    plt.figure(figsize=(20, 15))
    CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)

    plt.show()


def plot_3d(x, y, z):
    fig = plt.figure(figsize=(20, 15))
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    plt.show()
