"""
Generate heatmap type animations of the magnetic field around an area
written by Nathan Diggins (BooleanDesign)
"""
import numpy as np
import Magnetism as mag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


plt.rcParams['animation.ffmpeg_path'] = "C:\\ffmpeg\\bin\\ffmpeg.exe"
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)


def init_function():
    """
    Defines the initial character of the system
    """
    l = ax1.imshow(norm[:, :, 0], cmap=plt.cm.seismic)
    l.set_clim(cmax, cmin)
    ax1.set_aspect('equal', adjustable='box')
    print str(np.amin(z))
    ax1.set_title("Z=%s" % (str(np.amin(z))))
    fig1.colorbar(l, cax=cax)
    return l


def animate(i):
    cax.cla()
    l = ax1.imshow(norm[:, :, i], cmap=plt.cm.seismic)
    l.set_clim(cmax, cmin)
    ax1.set_title("Z=%s" % (str(np.amin(z) + (i * ((np.amax(z) - np.amin(z)) / frames)))))
    fig1.colorbar(l, cax=cax, extend='max')
    return l


"""
Generating the path
"""
n = 60
h = 15
t = np.linspace(0, np.pi * 2 * n, 500)
x = np.cos(t)
y = np.sin(t)
z = (h / (n * 2 * np.pi)) * t
path = mag.Path(x, y, z)

"""
Generating the field
"""

frames = 10
x_grid, y_grid, z_grid = np.meshgrid(np.linspace(-3, 3, 3),
                                     np.linspace(-3, 3, 3),
                                     np.linspace(-5, 20, frames))

w, v, u = path.mag_func(x_grid, y_grid, z_grid)

norm = np.sqrt(u ** 2 + v ** 2 + w ** 2)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
div = make_axes_locatable(ax1)
cax = div.append_axes('right', size='5%', pad=0.05)
plt.suptitle(r"$\parallel \vec{B} \parallel$ From Wire")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$y$")
cmin = np.amin(norm)
cmax = np.amax(norm)
ani = animation.FuncAnimation(fig1, animate, frames=frames, interval=100, init_func=init_function)
ani.save('test.mp4', writer=writer)
