import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import exists

import matplotlib.animation as animation

frame = 0
node = 0
while exists("data/" + str(frame) + "-" + str(node) + ".csv"):
    frame += 1
frame -= 1
while exists("data/" + str(frame) + "-" + str(node) + ".csv"):
    node += 1

fig, ax = plt.subplots()
size = 50
ax.set_xlim([-size, size])
ax.set_ylim([-size, size])


def get_points(i):
    x = []
    y = []
    for j in range(node):
        df = pd.read_csv("data/" + str(i) + "-" + str(j) + ".csv", header=None)
        x.extend(df[0].values.tolist())
        y.extend(df[1].values.tolist())
    return (x, y)


(x, y) = get_points(0)
scat = ax.scatter(x, y)


def animate(i):
    (x, y) = get_points(i)
    data = np.stack([x, y]).T
    scat.set_offsets(data)
    return (scat, )


ani = animation.FuncAnimation(fig,
                              animate,
                              repeat=True,
                              frames=135,
                              interval=50)

plt.show()
