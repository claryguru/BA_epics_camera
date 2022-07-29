import asyncio
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label, find_objects, median_filter
import scipy.optimize as opt
from sklearn.mixture import GaussianMixture


from math import pi

from image_aquirer import ImageAquirerFile, ImageAquirerVimba


class Image:
    def __init__(self, cam_dat_eps):
        self.cam_dat_eps = cam_dat_eps
        self.data = None
        self.edge = None
        self.update()

    def load_data(self):
        self.data = self.cam_dat_eps.get_image() #image is updated in a parallel loop

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

    def change_by_user(self, ctr_param_name, value):
        if ctr_param_name == 'roi_x_start':
            self.x_start = value
            print("set x_start param to", value)
        elif ctr_param_name == 'roi_y_start':
            self.y_start = value
            print("set y_start param to", value)
        elif ctr_param_name == 'roi_x_stop':
            self.x_stop = value
            print("set x_stop param to", value)
        elif ctr_param_name == 'roi_y_stop':
            self.y_stop = value
            print("set y_stop param to", value)

    def y_coordinate_in_image(self, y):
        return y + self.y_start

    def x_coordinate_in_image(self, x):
        return x + self.x_start

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class FitArea:
    def __init__(self, roi, factor, threshold, median=True, labeln=True):
        self.factor = factor
        self.threshold = threshold
        self.median = median
        self.labeln = labeln

        self.data = None
        self.edge = None
        self.pos_in_image_x = None
        self.pos_in_image_y = None

        self.update(roi)

    def update(self, roi):
        self.pos_in_image_x = roi.x_coordinate_in_image(0)
        self.pos_in_image_y = roi.y_coordinate_in_image(0)
        self.load_data(roi)
        self.edge = self.data.shape


    def load_data(self, roi):
        if self.labeln:
            #find frame around object, expand frame according to factor
            frame_expanded, med_filt_data, = self.label(roi)
            if frame_expanded:
                # slice roi to fit_area
                # if median is active, slice from median_filtered area
                if self.median:
                    self.data = med_filt_data[frame_expanded]
                else:
                    self.data = roi.data[frame_expanded]
            else:
                self.load_without_labeln(roi)
        else:
            self.load_without_labeln(roi)

    def load_without_labeln(self,roi):
            # if median is active, filter roi, otherwise fit_area data is roi
            if self.median:
                self.data = median_filter(roi.data, size=2)
            else:
                self.data = roi.data

    def label(self, roi):
        # median filter
        med_filt_data = median_filter(roi.data, size=2)

        # threshold filter
        thr_filt_data = med_filt_data > self.threshold

        # labeln
        labeled_data, num_label = label(thr_filt_data)
        if num_label > 1:
            print("found", num_label,  "Elektronenstrahl in roi")

        if num_label == 0:
            print("no labeld area found")
            self.data = roi.data
            frame_expanded = None
        else:
            # get frames around labeled objects
            frames = find_objects(labeled_data)

            # in case there a several frames, because there where more label than 1, choose the probaply biggest one
            if len(frames) > 1:
                frame = self.choose_frame(frames)
            else:
                frame = frames[0]

            # expand frame for certain factor but keep it in roi
            # the position in image is changed as well
            frame_expanded, roi_edge = self.expand_in_roi(frame, roi)

            # in case the edge of the roi was reached during expansion
            if roi_edge:
                print("Elektronenstrahl on the edge of the roi")

        return frame_expanded, med_filt_data

    def expand_in_roi(self, frame, roi):
        y_start, y_stop, x_start, x_stop = frame[0].start, frame[0].stop, frame[1].start, frame[1].stop
        add_x, add_y = self.calculate_expantion(self.factor, x_start, x_stop, y_start, y_stop)
        y_a_neu, y_e_neu, y_edge = self.expand_partly(add_y, y_start, y_stop, roi.edge[0])
        x_a_neu, x_e_neu, x_edge = self.expand_partly(add_x, x_start, x_stop, roi.edge[1])
        #change position in image
        self.pos_in_image_x = roi.x_coordinate_in_image(x_a_neu)
        self.pos_in_image_y = roi.y_coordinate_in_image(y_a_neu)
        return (slice(y_a_neu, y_e_neu, None), slice(x_a_neu, x_e_neu, None)), (y_edge or x_edge)

    def calculate_expantion(self, factor, x_start, x_stop, y_start, y_stop):
        dif_x = int(abs(x_stop - x_start) * factor)
        dif_y = int(abs(y_stop - y_start) * factor)
        return dif_x, dif_y

    def expand_partly(self, expantion, start, stop, max, min=0):
        start_new = start - expantion
        stop_new = stop + expantion
        edge = False
        if start_new < min:
            start_new = min
            edge = True
        if stop_new > max:
            stop_new = max
            edge = True
        return start_new, stop_new, edge

    def choose_frame(self, frames):
        print("chose frame")
        max_A = 0
        max_index = 0
        for index, frame in enumerate(frames):
            y_start, y_stop, x_start, x_stop = frame[0].start, frame[0].stop, frame[1].start, frame[1].stop
            A = (y_stop - y_start) * (x_stop - x_start)
            if A > max_A:
                max_A = A
                max_index = index
        return frames[max_index]

    def y_coordinate_in_image(self, y):
        return y + self.pos_in_image_y

    def x_coordinate_in_image(self, x):
        #double check ob außerhalb bild
        return x + self.pos_in_image_x

    def change_by_user(self, ctr_param_name, value):
        if ctr_param_name == 'factor':
            self.factor = value
            print("set factor param to", value)
        elif ctr_param_name == 'threshold':
            self.threshold = value
            print("set threshold param to", value)
        elif ctr_param_name == 'median_flt':
            self.median = value
            print("set median param to", value)

    def show(self):
        print(self.edge)
        print("position in image:", self.pos_in_image_x, self.pos_in_image_y)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class Gaussmodel:
    def __init__(self, sampled, fit_area, im):
        self.sampled = sampled
        self.initial_params = None
        self.result = None

        #create basis coordinates for later use
        edge_x = im.edge[1]
        edge_y = im.edge[0]
            # TO DO: double check if there is some better syntax
        self.x_values_basis = np.repeat(np.array([range(0, edge_x)]), edge_y, axis=0)
        y_list = []
        for i in range(0, edge_y):
            y_list.append([i] * edge_x)
        self.y_values_basis = np.array(y_list)

        #first fit analysis with founded guess and boundaries
        self.z_values_in = fit_area.data.flatten()
        self.x_values, self.y_values = self.build_xy_values(fit_area)
        self.initial_fit(self.x_values, self.y_values, self.z_values_in)

    def initial_fit(self, x, y, z):
        initial_params = self.guess(x,y,z)
        bounds = ([0, 0, 0, 0, 0, -pi / 2, 0], [16382, np.inf, np.inf, np.inf, np.inf, pi / 2, 16382])
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (x, y), z, p0=initial_params, bounds=bounds)
        [amplitude, centerx, centery, sigmax, sigmay, rot, offset] = popt.tolist()
        self.initial_params = amplitude, centerx, centery, sigmax, sigmay, rot, offset
        self.result = popt

    def build_xy_values(self, fit_area):
        edge_x = fit_area.edge[1]
        edge_y = fit_area.edge[0]
        slice_x_y = (slice(0, edge_y, None), slice(0, edge_x, None))
        x_values = self.x_values_basis[slice_x_y].flatten()
        y_values = self.y_values_basis[slice_x_y].flatten()
        return x_values, y_values

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
        #guess the approximate values for the first fit analysis
        maxx, minx = np.amax(x), np.amin(x)
        maxy, miny = np.amax(y), np.amin(y)
        maxz, minz = np.amax(z), np.amin(z)

        centerx = x[np.argmax(z)]
        centery = y[np.argmax(z)]
        amplitude = (maxz - minz)  # quasi height

        # stimmt wenn gauss eng am Bildrand ist
        # vielleicht über objekt größe bestimmen
        sigmax = (maxx - minx) / 6.0
        sigmay = (maxy - miny) / 6.0
        offset = minz

        return amplitude, centerx, centery, sigmax, sigmay, 0, offset

    def update(self, fit_area):
        self.z_values_in = fit_area.data.flatten()
        self.x_values, self.y_values = self.build_xy_values(fit_area)

        #sample values if samped is active
        if self.sampled > 1:
            self.z_values_in = self.z_values_in[0:(len(self.z_values_in)):self.sampled]
            self.x_values = self.x_values[0:(len(self.x_values)):self.sampled]
            self.y_values = self.y_values[0:(len(self.y_values)):self.sampled]

        self.initial_params = self.result
        popt, pcov = opt.curve_fit(self.twoD_Gaussian, (self.x_values, self.y_values), self.z_values_in, p0=self.initial_params)
        self.result = popt

    def get_params(self,fit_area):
        [amplitude, centerx, centery, sigmax, sigmay, rot, offset] = self.result.tolist()
        #centerx and centery are so far in the fit_area coordinates and have to be projected to image
        return [amplitude, fit_area.x_coordinate_in_image(centerx), fit_area.y_coordinate_in_image(centery),
                sigmax, sigmay, rot, offset]

    def change_by_user(self, ctr_param_name, value):
        if ctr_param_name == 'sampled':
            self.sampled = value
            print("set sampeled param to", value)

    def show(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # old data
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("data in")
        ax.scatter(self.x_values, self.y_values, self.z_values_in, c=self.z_values_in, cmap='viridis', linewidth=0.5)

        # fitted data
        print(self.result)
        z_values_new = self.twoD_Gaussian((self.x_values, self.y_values), *(self.result))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title("data fitted")
        ax.scatter(self.x_values, self.y_values, z_values_new, c=z_values_new, cmap='viridis', linewidth=0.5)

        plt.show()


class DataAnalyzer:
    def __init__(self, cam_dat_eps, init_dict=None):
        self.cam_dat_eps = cam_dat_eps
        self.im = Image(self.cam_dat_eps)

        self.control_params = self.load_control_params(init_dict)
        #init values (same default values as in epics interface)
        self.roi = Roi(self.control_params['roi_x_start'], self.control_params['roi_x_stop'],
                       self.control_params['roi_y_start'], self.control_params['roi_y_stop'], self.im)
        self.fit_area = FitArea(self.roi, self.control_params['factor'],
                                self.control_params['threshold'], self.control_params['median_flt'])
        self.g_model = Gaussmodel(self.control_params['sampled'], self.fit_area, self.im)
        self.params = self.g_model.get_params(self.fit_area)
        print("first params berechnet ", self.params)

    def load_control_params(self, init_dict):
        default_control_param_values = \
                {'roi_x_start': 0,
                'roi_x_stop': self.im.edge[1],
                'roi_y_start': 0,
                'roi_y_stop': self.im.edge[0],
                'factor': 0.1,
                'threshold': self.define_threshold(),
                'median_flt': True,
                'sampled': 1}

        control_param_values = default_control_param_values

        if init_dict:
            if 'control_params_values' in init_dict:
                for param, value in init_dict['control_params_values'].items():
                    control_param_values[param] = value

        return control_param_values

    def define_threshold(self):
        classif = GaussianMixture(n_components=1)
        classif.fit(self.im.data.reshape((self.im.data.size, 1)))
        threshold = np.mean(classif.means_)
        return threshold


    def get_init_control_params(self):
        return self.control_params

    def get_data_a_settings(self):
        return{'control_params_values': self.control_params}

    async def analyze(self):
        while True:
            self.im.update()
            self.roi.update(self.im)
            self.fit_area.update(self.roi)
            self.g_model.update(self.fit_area)
            self.params = self.g_model.get_params(self.fit_area)
            print("params berechnet ", self.params)
            await asyncio.sleep(2)

    def analyze_sync(self):
        self.im.update()
        self.roi.update(self.im)
        self.fit_area.update(self.roi)
        self.g_model.update(self.fit_area)
        self.params = self.g_model.get_params(self.fit_area)
        print("params berechnet ", self.params)

    def change_by_user(self, area, control_param_name, value):
        if area == 'roi':
            self.roi.change_by_user(control_param_name, value)
        elif area == 'fit_area':
            self.fit_area.change_by_user(control_param_name, value)
        elif area == 'g_model':
            self.fit_area.change_by_user(control_param_name, value)
        else:
            print("There is no such area to update")

        #update for later use
        self.control_params[control_param_name] = value

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
    example_init_rot = {'control_params_values':
                {'roi_x_start': 1000,
                'roi_x_stop': 1500,
                'roi_y_start': 750,
                'roi_y_stop': 1200,
                'factor': 1,
                'threshold': 708,
                'median_flt': True,
                'sampled': 1}}

    example_init = {'control_params_values':
                    {'roi_x_start': 800,
                    'roi_x_stop': 1250,
                    'roi_y_start': 600,
                    'roi_y_stop': 900,
                    'factor': 0.1,
                    'threshold': 708,
                    'median_flt': True,
                    'sampled': 1}}

    class fake_CamDatEps:
        def __init__(self):
            #self.ia = ImageAquirerFile(self, 'D:\\HZB\\Camera_Data\\mls13\\', 200)
            self.ia = ImageAquirerVimba(fake_CamDatEps, {"features": {"ExposureAuto":"Off"}})
            self.data_analyzer = DataAnalyzer(self, example_init)
            self.data_analyzer.show()

            for i in range(0, 10):
                #self.ia.aquire_sync()
                self.ia.get_frame()
                self.data_analyzer.analyze_sync()

        def get_image(self):
            return self.ia.get_image()

    f_camdatep = fake_CamDatEps()