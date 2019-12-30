"""
Generate heatmap type animations of the magnetic field around an area
written by Nathan Diggins (BooleanDesign)
"""
import numpy as np
import Magnetism as mag
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


def init_function():
    """
    Defines the initial character of the system
    """
    l = ax1.imshow(norm[:, :, 0])
    ax1.set_aspect('equal', adjustable='box')
    print str(np.amin(z))
    ax1.set_title("Z=%s" % (str(np.amin(z))))
    return l


def animate(i):
    l = ax1.imshow(norm[:, :, i])
    ax1.set_title("Z=%s" % (str(np.amin(z) + (i * ((np.amax(z) - np.amin(z)) / frames)))))
    return l


"""
Generating the path
"""
t = np.linspace(-5, 10, 200)
x = 0 * t
y = 0 * t
z = t
path = mag.Path(x, y, z)

"""
Generating the field
"""

frames = 100
x_grid, y_grid, z_grid = np.meshgrid(np.linspace(-3, 3, 10),
                                     np.linspace(-3, 3, 10),
                                     np.linspace(-10, 15, frames))

w, v, u = path.mag_func(x_grid, y_grid, z_grid)

norm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

fig1 = plt.figure()
l = ax1 = fig1.add_subplot(111)
ani = animation.FuncAnimation(fig1, animate, frames=frames, interval=100, init_func=init_function)
ani.save('test.mp4', writer=writer)
