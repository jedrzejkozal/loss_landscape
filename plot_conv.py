from plotting.ploting_points import *
from models.conv import conv_model
from datasets import mnist_dataset

import matplotlib.cm as cm
import matplotlib.pyplot as plt


dataset = mnist_dataset()
model = conv_model(dataset)
x, y, z = get_ploting_points(model, dataset[2], dataset[3])


fig, ax = plt.subplots()
CS = ax.contour(x, y, z)
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Simplest default with labels')

plt.show()
