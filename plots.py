import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d(x, y, figure_index=0, label=None):
    plt.figure(num=figure_index, figsize=(20, 15))
    plt.plot(x, y, '-', label=label)
    if label is not None:
        plt.legend()


def plot_levels(x, y, z, figure_index=0):
    plt.figure(num=figure_index, figsize=(20, 15))
    CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.savefig('levels.png')


def plot_3d(x, y, z, figure_index=0):
    fig = plt.figure(num=figure_index, figsize=(20, 15))
    ax = Axes3D(fig)
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    plt.savefig('plot_3d.png')


def show_all():
    # plt.show()
    plt.savefig('fig.png')
