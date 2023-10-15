import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.animation as animation

fig, ax = plt.subplots()
size = 4
ax.set_xlim([-size, size])
ax.set_ylim([-size, size])


def get_points(i):
    x = []
    y = []
    for j in range(8):
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
                              frames=32,
                              interval=50)

plt.show()
