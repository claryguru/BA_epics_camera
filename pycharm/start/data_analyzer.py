import asyncio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label, find_objects, median_filter
import scipy.optimize as opt
from math import e, sqrt, pi

from image_aquirer import ImageAquirerFile, ImageAquirerVimba


class Image:
    def __init__(self, image_aquirer):
        self.ia = image_aquirer

        self.data = None
        self.edge = None
        self.update()

    def load_data(self):
        self.data = self.ia.get_image() #image is updated in a parallel loop

    def update(self):
        self.load_data()
        self.edge = self.data.shape

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class Roi:
    def __init__(self, x_start, x_stop, y_start, y_stop, image):
        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop

        self.data = None
        self.edge = None
        self.update(image)

    def load_data(self, image):
        self.data = image.data[(slice(self.y_start, self.y_stop, None),
                                slice(self.x_start, self.x_stop, None))]

    def update(self, image):
        self.load_data(image)
        self.edge = self.data.shape

    def change_position_by_user(self, x_start, x_stop, y_start, y_stop):
        pass

    def y_coordinate_in_image(self, y):
        return y + self.y_start

    def x_coordinate_in_image(self, x):
        return x + self.x_start

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class FitArea:
    def __init__(self, roi, image, factor, threshold, median=True):
        self.factor = factor
        self.threshold = threshold
        self.median = median

        self.data = None
        self.edge = None
        self.update(roi, image)

    def load_data(self, roi, image):
        # median filter
        data = roi.data
        if self.median:
            data = median_filter(data, size=2)

        # threshold filter
        data = data > self.threshold

        # labeln
        data, num_label = label(data)
        # TO DO: check num_label with error

        # fit data frame
        frame = find_objects(data)
        # TO DO: check if only one object (sonst geht frame[0] nicht)
        y_start, y_stop, x_start, x_stop = frame[0][0].start, frame[0][0].stop, frame[0][1].start, frame[0][1].stop
        dif_x, dif_y = self.calculate_expantion(self.factor, x_start, x_stop, y_start, y_stop)
        frame = self.expand(dif_x, dif_y,
                            roi.x_coordinate_in_image(x_start), roi.x_coordinate_in_image(x_stop),
                            roi.y_coordinate_in_image(y_start), roi.y_coordinate_in_image(y_stop),
                            image.edge[0], image.edge[1])
        # TO DO: check if max werte stimmen

        # slice image for
        self.data = image.data[frame]

    def calculate_expantion(self, factor, x_start, x_stop, y_start, y_stop):
        dif_x = int(abs(x_stop - x_start) * factor)
        dif_y = int(abs(y_stop - y_start) * factor)
        return dif_x, dif_y

    def expand(self, add_x, add_y, x_start, x_stop, y_start, y_stop, max_x, max_y):
        y_a_neu, y_e_neu = self.expand_partly(add_y, y_start, y_stop, max_y)
        x_a_neu, x_e_neu = self.expand_partly(add_x, x_start, x_stop, max_x)
        return (slice(y_a_neu, y_e_neu, None), slice(x_a_neu, x_e_neu, None))

    def expand_partly(self, expantion, start, stop, max, min=0):
        start_new = start - expantion
        stop_new = stop + expantion
        if start_new < min:
            start_new = min
        if stop_new > max:
            stop_new = max
        return start_new, stop_new

    def update(self, roi, image):
        self.load_data(roi, image)
        self.edge = self.data.shape

    def change_factor_by_user(self):
        pass

    def change_threshold_by_user(self):
        pass

    def change_median_usage_by_user(self):
        pass

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class Gaussmodel:
    def __init__(self, fit_area):
        self.z_values_in = fit_area.data.flatten()
        edge_x = fit_area.edge[1]
        edge_y = fit_area.edge[0]
        # TO DO: double check if there is some better syntax
        self.x_values = np.repeat(np.array([range(0, edge_x)]), edge_y, axis=0).flatten()
        self.y_values = np.repeat(np.array([range(edge_y - 1, -1, -1)]).reshape(edge_y, 1), edge_x, axis=1).flatten()

        self.initial_params = self.guess(self.x_values, self.y_values, self.z_values_in)
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (self.x_values, self.y_values), self.z_values_in, p0=self.initial_params)
        self.result = popt

    def twoD_Gaussian(self, x_y, amplitude, x_center, y_center, sigma_x, sigma_y, theta, offset):
        x, y = x_y
        xo = float(x_center)
        yo = float(y_center)
        a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
        b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
        c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
        result = offset + amplitude * np.exp(
            - (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
        return result.ravel()

    def guess(self, x, y, z):
        """Estimate starting values from 2D peak data and create Parameters."""
        if x is None or y is None:
            return 1.0, 0.0, 0.0, 1.0, 1.0

        maxx, minx = np.amax(x), np.amin(x)
        maxy, miny = np.amax(y), np.amin(y)
        maxz, minz = np.amax(z), np.amin(z)

        centerx = x[np.argmax(z)]
        centery = y[np.argmax(z)]
        amplitude = (maxz - minz)  # quasi height
        sigmax = (maxx - minx) / 6.0
        sigmay = (maxy - miny) / 6.0
        offset = minz

        return amplitude, centerx, centery, sigmax, sigmay, 0, offset

    def update(self, fit_area):
        self.z_values_in = fit_area.data.flatten()
        edge_x = fit_area.edge[1]
        edge_y = fit_area.edge[0]
        # TO DO: double check if there is some better syntax
        self.x_values = np.repeat(np.array([range(0, edge_x)]), edge_y, axis=0).flatten()
        self.y_values = np.repeat(np.array([range(edge_y - 1, -1, -1)]).reshape(edge_y, 1), edge_x, axis=1).flatten()

        self.initial_params = self.get_result()
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (self.x_values, self.y_values), self.z_values_in,
                                   p0=self.initial_params)
        self.result = popt

    def get_result(self):
        [amplitude, centerx, centery, sigmax, sigmay, rot, offset] = self.result.tolist()
        return amplitude, centerx, centery, sigmax, sigmay, rot, offset

    def get_params(self):
        return self.result.tolist()

    def show(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # old data
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("data in")
        ax.scatter(self.x_values, self.y_values, self.z_values_in, c=self.z_values_in, cmap='viridis', linewidth=0.5)

        # fitted data
        print(self.get_result())
        z_values_new = self.twoD_Gaussian((self.x_values, self.y_values), *(self.get_result()))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title("data fitted")
        ax.scatter(self.x_values, self.y_values, z_values_new, c=z_values_new, cmap='viridis', linewidth=0.5)

        plt.show()


class DataAnalyzer:
    def __init__(self, ia):
        self.ia = ia #ImageAquirer('D:\\HZB\\Camera_Data\\test_roi\\')
        self.im = Image(self.ia)
        self.roi = Roi(800, 1250, 600, 900, self.im)
        self.fit_area = FitArea(self.roi, self.im, 0.5, 708)
        self.g_model = Gaussmodel(self.fit_area)
        self.params = self.g_model.get_params()
        print("params berechnet ", self.params)

    async def analyze(self):
        while True:
            self.im.update()
            self.roi.update(self.im)
            self.fit_area.update(self.roi, self.im)
            self.g_model.update(self.fit_area)
            self.params= self.g_model.get_params()
            print("params berechnet ", self.params)
            await asyncio.sleep(0)

    def analyze_sync(self):
        self.im.update()
        self.roi.update(self.im)
        self.fit_area.update(self.roi, self.im)
        self.g_model.update(self.fit_area)
        self.params = self.g_model.get_params()
        print("params berechnet ", self.params)

    def show(self):
        print("im")
        self.im.show()
        print("roi")
        self.roi.show()
        print("fit_area")
        self.fit_area.show()
        print("model")
        self.g_model.show()

if __name__ == '__main__':
    ia = ImageAquirerFile('D:\\HZB\\Camera_Data\\mls13\\', 200)
    data_analyzer = DataAnalyzer(ia)
    data_analyzer.show()

    for i in range(0, 10):
        ia.aquire_sync()
        data_analyzer.analyze_sync()