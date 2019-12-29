"""
Generate heatmap type animations of the magnetic field around an area
written by Nathan Diggins (BooleanDesign)
"""

import numpy as np
import Magnetism as mag
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def init_function():
    """
    Defines the initial character of the system
    """
    l = ax1.imshow(norm[::][0])
    return l

def animate(i):
    l = ax1.imshow(norm[::][i])
    return l

"""
Generating the path
"""
t = np.linspace(0,np.pi*2,5000)
x = np.cos(t)
y = np.sin(t)
z = 0*t
path = mag.Path(x,y,z)

"""
Generating the field
"""
x_grid, y_grid, z_grid = np.meshgrid(np.linspace(-3,3,10),
                                     np.linspace(-3,3,10),
                                     np.linspace(-3,3,25))

u,v,w = path.mag_func(x_grid,y_grid,z_grid)

norm = np.sqrt(u**2+v**2+w**2)

fig1 = plt.figure()
l = ax1 = fig1.add_subplot(111)
ani = FuncAnimation(fig1,animate,frames=np.arange(0,100,1),init_func=init_function)
plt.show()