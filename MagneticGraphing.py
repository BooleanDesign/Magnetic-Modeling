"""

Quiver Plotting system for magnetic fields
Written by: Nathan Diggins

"""
"""
GENERAL TODOS
"""
# TODO: Create a


import Magnetism as mag
import numpy as np
import matplotlib.pyplot as plt  # For graphing capabilities
from mpl_toolkits.mplot3d import Axes3D  # For 3-D graphing capabilities

"""
Defining the paths, can be altered in any way desired
"""
t = np.linspace(-5, 5, 100)
xs = 0 * t
ys = 0 * t
zs = t
path = mag.Path(xs, ys, zs)

"""
Creating the graph data
"""

x, y, z = np.meshgrid(np.linspace(-2, 2, 5),
                      np.linspace(-2, 2, 5),
                      np.linspace(-5, 5, 15))

u, v, w = path.mag_func(x, y, z)

"""
Creating the plot
"""

figure = plt.figure()
ax1 = figure.add_subplot(111, projection="3d")
norm = np.sqrt(u ** 2 + v ** 2 + w ** 2)
plot = ax1.quiver(x, y, z, u / norm, v / norm, w / norm, )
plt.show()
