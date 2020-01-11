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


def get_file_path(file_path, config_path=None):
    """
    Reads a path object from the file specified
    :param file_path: <str> specified route to the file
    :param config_path: <str> config file, None by default
    :return: <Path> based on file reading
    """
    if config_path is not None:  # Assuming that there is a custom config file, the data has to be extracted
        try:  # Opening the config file
            config_file = open(config_path, 'r+')
        except:  # If the progam cannot find the configuration file.
            raise IOError("Failed to find config file: %s." % (str(config_path)))
        try:  # This is formatting the configuration data properly
            config_data = {i.split("=")[0]: i.split("=")[1] for i in config_file.read().split('\n')}
        except:
            raise SyntaxError("Config data was the incorrect form.")
        for setting in config_data:
            # This should attempt to form the settings as floats if possible
            try:
                config_data['setting'] = float(config_data['setting'])
            except:
                pass
    else:
        config_data = {'current': 1.0}
    """
    Now builds the actual path from the specified file
    """
    try:
        path_file = open(file_path, 'r+')
    except IOError:
        raise IOError("Failed to find %s." % (str(file_path)))
    try:
        path_data = [(float(i.split(',')[0]), float(i.split(',')[1]), float(i.split(',')[2])) for i in
                     path_file.read().split('\n')]
        xs = [i[0] for i in path_data]  # Defines the x,y,z portions of the path
        ys = [i[1] for i in path_data]
        zs = [i[2] for i in path_data]
        # TODO: Check efficiency on this section of code.
    except:
        raise SyntaxError("Path file was not the correct format.")

    path_file.close()
    return Path(xs, ys, zs, current=config_data['current'])


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
