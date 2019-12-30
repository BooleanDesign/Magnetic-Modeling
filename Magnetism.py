import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os

base_config = {}


def vector_length(x, y, z):
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def normal_vector(x, y, z):
    radius = vector_length(x, y, z)
    return np.stack([x / radius, y / radius, z / radius], axis=-1)


class Path:
    """
    This defines the Path class which allows for the calculations of the magnetic field.
    """

    def __init__(self, xs, ys, zs, current=1.0):
        self.points = zip(*[xs, ys, zs])  # defines the points
        self.x = xs
        self.y = ys
        self.z = zs
        self.current = current
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

    def mag_func(self, x, y, z, mag_const=1.25663706212e-6):
        """
        Generates the magnetic field function for the class.
        :param x: np.meshgrid
        :param y: np.meshgrid
        :param z: np.meshgrid
        :param mag_const: Mu_naut
        :return: Returns meshgrid of vector form Biot-Savart
        """
        # TODO: Better comments throughout the function
        mag_param = self.current * mag_const / (4 * np.pi)
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
