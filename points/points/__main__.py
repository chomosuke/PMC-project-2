import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.animation as animation

fig, ax = plt.subplots()
ax.set_xlim([-10, 10])
ax.set_ylim([-10, 10])


def get_points(i):
    x = []
    y = []
    for j in range(6):
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
                              frames=99,
                              interval=50)

plt.show()
