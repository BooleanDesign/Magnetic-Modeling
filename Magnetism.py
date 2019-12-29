import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def vector_length(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def normal_vector(x, y, z):
    radius = vector_length(x, y, z)
    return np.stack([x / radius, y / radius, z / radius], axis=-1)


class Path:
    """
    This defines the Path class which allows for the calculations of the magnetic field.
    """

    def __init__(self, xs, ys, zs):
        self.points = zip(*[xs, ys, zs])  # defines the points
        self.x = xs
        self.y = ys
        self.z = zs
        self.path_vectors = [(self.points[i + 1][0] - self.points[i][0],
                              self.points[i + 1][1] - self.points[i][1],
                              self.points[i + 1][2] - self.points[i][2]) for i in range(len(self.x) - 1)]

    def get_length(self):
        """
        Calculates the path length
        :return: returns float length
        """
        return sum([np.sqrt(((self.x[i + 1] - self.x[i]) ** 2) + ((self.y[i + 1] - self.y[i]) ** 2) + (
                (self.z[i + 1] - self.z[i]) ** 2)) for i in
                    range(len(self.x) - 1)])

    def mag_func(self, x, y, z, current=1.0, mag_const=1.25663706212e-6):
        mag_param = current * mag_const / (4 * np.pi)
        s = x.shape
        res = np.zeros((s[0], s[1], s[2], 3))
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    print str(100 * ((float(k) / float(s[0] * s[1] * s[2])) + (float(j) / float(s[0] * s[1])) + (
                            float(i) / float(s[0])))) + "%"
                    for idx, (xc, yc, zc) in enumerate(zip(self.x[:-1], self.y[:-1], self.z[:-1])):
                        res[i, j, k, :] += mag_param * \
                                           np.cross(self.path_vectors[idx], [x[i, j, k] - xc,
                                                                             y[i, j, k] - yc, z[i, j, k] - zc]) / \
                                           np.linalg.norm([x[i, j, k] - xc, y[i, j, k] - yc,
                                                           z[i, j, k] - zc]) ** 2
        return res[:, :, :, 0], res[:, :, :, 1], res[:, :, :, 2]


n = 200
r = 6
h = 30
grid_x, grid_y, grid_z = np.meshgrid(np.linspace(-5, 5, 5),
                                     np.linspace(-5, 5, 5),
                                     np.linspace(-40, 40, 10))
c = h / (2 * n * np.pi)
t = np.linspace(0, 2 * n * np.pi, 5000)
xp = r * np.cos(t)
yp = r * np.sin(t)
zp = (h / (2 * np.pi * n)) * t - 15
p = Path(list(xp), list(yp), list(zp))
func = p.mag_func(grid_x, grid_y, grid_z)
u, v, w = p.mag_func(grid_x, grid_y, grid_z)
r = np.sqrt(u ** 2 + v ** 2 + w ** 2)
ax1 = plt.subplot(111, projection='3d')
ax1.plot(xp, yp, zp, 'r-')
ax1.quiver(grid_x, grid_y, grid_z, u / r, v / r, w / r, length=2)
plt.show()
