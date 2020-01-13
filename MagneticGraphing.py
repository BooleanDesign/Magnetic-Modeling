"""
Magnetic Streamplot Grapher
Written by: Nathan Diggins
"""

import Magnetism as mag
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


def initial_function():
    """
    Plotting the slices along the given axis
    :param plane: plane to construct the splicing along
    :type plane: str
    :return: ax object
    """
    initial_x, initial_y = x[:, :, 0], y[:, :, 0]
    angles = np.arcsin(w[:, :, 0] / norm[:, :, 0])
    initial_u, initial_v = u[:, :, 0], v[:, :, 0]
    plot = ax1.streamplot(initial_x, initial_y, initial_u, initial_v)
    return plot


def animate(i):
    """
    Animates the graphs
    :return: ax object
    """
    ax1.cla()
    ani_x, ani_y = x[:, :, i], y[:, :, i]
    angles = np.arcsin(w[:, :, i] / norm[:, :, i])
    ani_u, ani_v = u[:, :, i] * np.cos(angles), v[:, :, i] * np.cos(angles)
    plot = ax1.streamplot(ani_x, ani_y, ani_u, ani_v)
    return plot


"""
Generate the data
"""
frames = 100
x, y, z = np.meshgrid(np.linspace(-5, 5, 10),
                      np.linspace(-5, 5, 10),
                      np.linspace(-5, 5, frames))

t = np.linspace(0, np.pi * 2, 100)
magnetic_path = mag.Path(0 * t, 2 * np.cos(t), 2 * np.sin(t))

u, v, w = magnetic_path.mag_func(x, y, z)
norm = np.sqrt(u ** 2 + v ** 2 + w ** 2)
"""
Constructing the plots
"""

figure1 = plt.figure()
ax1 = figure1.add_subplot(111)
plt.suptitle(r"$\parallel \vec{B} \parallel$ From Wire")
ani = animation.FuncAnimation(figure1, animate, frames=frames, interval=100, init_func=initial_function)
ani.save('streamtest.mp4', writer=writer)
