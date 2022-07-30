import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label, find_objects, median_filter
import scipy.optimize as opt
from sklearn.mixture import GaussianMixture
from math import pi

from image_aquirer import ImageAquirerFile, ImageAquirerVimba


class Image:
    '''
    --- data image ---
    biggest data loads data from cam_dat_eps connection
    if no image was loaded 'image.data = None'
    '''
    def __init__(self, cam_dat_eps):
        self.cam_dat_eps = cam_dat_eps

        self.data = None
        self.edge = None
        self.update()

    def load_data(self):
        self.data = self.cam_dat_eps.get_image() # image is updated in a parallel loop

    def update(self):
        self.load_data()

        # check if there is a current image data
        if self.data is not None:
            self.edge = self.data.shape
        else:
            error_message = 'loading data to image for analyzing failed'
            print(error_message)
            self.cam_dat_eps.set_da_error(error_message)

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class Roi:
    '''
    --- data region of interest ---
    image is sliced to smaller part by user defined values creating the roi
    x_start, x_stop, y_start, y_stop - user defined slices for creating roi
    default: roi is as big as image
    '''
    def __init__(self, x_start, x_stop, y_start, y_stop, cam_dat_eps):
        self.cam_dat_eps = cam_dat_eps

        self.x_start = x_start
        self.x_stop = x_stop
        self.y_start = y_start
        self.y_stop = y_stop

        self.data = None
        self.edge = None

    def init(self, image):
            self.update(image)

    def load_data(self, image):
            self.data = image.data[(slice(self.y_start, self.y_stop, None),
                                slice(self.x_start, self.x_stop, None))]

    def update(self, image):
        try:
            self.load_data(image)
            self.edge = self.data.shape
        except:
            error_message = 'loading data from image to create roi failed, maybe check roi parameters'
            self.send_error(error_message)
            self.data = None
            self.edge = None

    def change_by_user(self, control_param_name, value):
        if control_param_name == 'roi_x_start':
            self.x_start = value
            print("set x_start param to", value)
        elif control_param_name == 'roi_y_start':
            self.y_start = value
            print("set y_start param to", value)
        elif control_param_name == 'roi_x_stop':
            self.x_stop = value
            print("set x_stop param to", value)
        elif control_param_name == 'roi_y_stop':
            self.y_stop = value
            print("set y_stop param to", value)
        else:
            error_message = 'changing roi control parameter failed'
            self.send_error(error_message)

    def y_coordinate_in_image(self, y):
        try:
            return y + self.y_start
        except:
            error_message = 'finding y-coordinate in image unsuccessful'
            self.send_error(error_message)
            return y

    def x_coordinate_in_image(self, x):
        try:
            return x + self.x_start
        except:
            error_message = 'finding x-coordinate in image unsuccessful'
            self.send_error(error_message)
            return x

    def send_error(self, error_message):
        print(error_message)
        self.cam_dat_eps.set_da_error(error_message)

    def show(self):
        print(self.edge)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class FitArea:
    '''
    --- data fit data ---
    data used by fit algorithm to analyse data
    created by finding electron object in roi and slicing a frame around it
    labeln: if False fit data is the same as roi, otherwise object is found by labeling areas divide by threshold
    threshold: decides if data value could be electron obeject or not, default is calculated by gaussian Mixture Model,
              otherwise defined by user, might be useful to change, if there a many or none labeled objects found in roi
    factor: expand frame around object by factor, x/y length * factor is the additional expansion
    median: in order to make the object labeling work, the data is filtered with a median filter
              since filtering the data for eliminating pixel is useful anyway, the user can define if the filtered roi
              is used for slicing the fit data as well
    '''
    def __init__(self, factor, threshold, cam_dat_eps, median=True, labeln=True):
        self.cam_dat_eps = cam_dat_eps

        self.factor = factor
        self.threshold = threshold
        self.median = median
        self.labeln = labeln

        self.data = None
        self.edge = None
        self.pos_in_image_x = None
        self.pos_in_image_y = None

    def init(self, roi):
        self.update(roi)

    def update(self, roi):
        try:
            self.pos_in_image_x = roi.x_coordinate_in_image(0)
            self.pos_in_image_y = roi.y_coordinate_in_image(0)
            self.load_data(roi)
            self.edge = self.data.shape
        except:
            error_message = 'creating fit_area from roi failed'
            self.send_error(error_message)

    def load_data(self, roi):
        if self.labeln:
            # find frame around object, expand frame according to factor
            frame_expanded, med_filt_data, = self.label(roi)
            if frame_expanded:
                # slice roi to fit_area
                # if median is active, slice from median_filtered data
                if self.median:
                    self.data = med_filt_data[frame_expanded]
                else:
                    self.data = roi.data[frame_expanded]
            else:
                self.load_without_labeln(roi)
        else:
            self.load_without_labeln(roi)

    def load_without_labeln(self, roi):
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
            error_message = 'found '+num_label+' possible electron objects in roi, maybe change threshold'
            self.send_error(error_message)

        if num_label == 0:
            error_message = 'found no possible electron objects in roi, maybe change threshold'
            self.send_error(error_message)

            self.data = roi.data
            frame_expanded = None

        else:
            # get frames around labeled objects
            frames = find_objects(labeled_data)

            # if there a several frames, because there where more label than 1, choose the one with the biggest data
            if len(frames) > 1:
                frame = self.choose_frame(frames)
            else:
                frame = frames[0]

            # expand frame for certain factor but keep it in roi
            # the position in image is changed as well
            frame_expanded, roi_edge = self.expand_in_roi(frame, roi)

            # in case the edge of the roi was reached during expansion
            if roi_edge:
                error_message = 'electron object is at the edge of roi, maybe change roi'
                self.send_error(error_message)

        return frame_expanded, med_filt_data

    def expand_in_roi(self, frame, roi):
        # calculate how much is added in x and y direction and add it to frame
        # keep it in roi
        y_start, y_stop, x_start, x_stop = frame[0].start, frame[0].stop, frame[1].start, frame[1].stop
        add_x, add_y = self.calculate_expansion(self.factor, x_start, x_stop, y_start, y_stop)
        y_a_neu, y_e_neu, y_edge = self.expand_partly(add_y, y_start, y_stop, roi.edge[0])
        x_a_neu, x_e_neu, x_edge = self.expand_partly(add_x, x_start, x_stop, roi.edge[1])

        # change position in image
        self.pos_in_image_x = roi.x_coordinate_in_image(x_a_neu)
        self.pos_in_image_y = roi.y_coordinate_in_image(y_a_neu)

        return (slice(y_a_neu, y_e_neu, None), slice(x_a_neu, x_e_neu, None)), (y_edge or x_edge)

    def calculate_expansion(self, factor, x_start, x_stop, y_start, y_stop):
        # use factor to calculate expansion
        try:
            dif_x = int(abs(x_stop - x_start) * factor)
            dif_y = int(abs(y_stop - y_start) * factor)
            return dif_x, dif_y
        except:
            error_message = 'calculate expansion unsuccessful'
            self.send_error(error_message)
            return 0, 0

    def expand_partly(self, expansion, start, stop, max, min=0):
        # expand frame values in x and y direction
        # keep them in value range of min and max of the roi
        try:
            start_new = start - expansion
            stop_new = stop + expansion
            edge = False
            if start_new < min:
                start_new = min
                edge = True
            if stop_new > max:
                stop_new = max
                edge = True
            return start_new, stop_new, edge
        except:
            error_message = ' expanding electron object frame unsuccessful'
            self.send_error(error_message)
            return start, stop, False

    def choose_frame(self, frames):
        # choose frame by biggest data
        # TO DO: find a better more physically correct solution for finding the right object
        try:
            max_A = 0
            max_index = 0
            for index, frame in enumerate(frames):
                y_start, y_stop, x_start, x_stop = frame[0].start, frame[0].stop, frame[1].start, frame[1].stop
                A = (y_stop - y_start) * (x_stop - x_start)
                if A > max_A:
                    max_A = A
                    max_index = index
            return frames[max_index]
        except:
            error_message = 'choosing frame of electron object unsuccessful'
            self.send_error(error_message)
            return frames[0]

    def y_coordinate_in_image(self, y):
        try:
            return y + self.pos_in_image_y
        except:
            error_message = 'finding y-coordinate in image unsuccessful'
            self.send_error(error_message)
            return y

    def x_coordinate_in_image(self, x):
        try:
            return x + self.pos_in_image_x
        except:
            error_message = 'finding x-coordinate in image unsuccessful'
            self.send_error(error_message)
            return x

    def change_by_user(self, control_param_name, value):
        if control_param_name == 'factor':
            self.factor = value
            print("set factor param to", value)
        elif control_param_name == 'threshold':
            self.threshold = value
            print("set threshold param to", value)
        elif control_param_name == 'median_flt':
            self.median = value
            print("set median param to", value)
        else:
            error_message = 'changing fit data control parameter failed'
            self.send_error(error_message)

    def send_error(self, error_message):
        print(error_message)
        self.cam_dat_eps.set_da_error(error_message)

    def show(self):
        print(self.edge)
        print("position in image:", self.pos_in_image_x, self.pos_in_image_y)
        plt.imshow(self.data, cmap='gray')
        plt.show()


class GaussianFit:
    '''
    --- gaussian fit ---
    contains model and fit algorithm for analysis
    data in for fit algorithm -> created x and y data, fit data data
    used fit algorithm -> opt.curve_fit, other algorithms would be possible
    sampled: int n, sample each nth element from fit data, n=1 means no sampling and is the default value
    initial_params: fit params from last analysis
    '''
    def __init__(self, sampled, cam_dat_eps):
        self.cam_dat_eps = cam_dat_eps

        self.sampled = sampled
        self.initial_params = None
        self.result = None

        # initialized once for later use
        self.x_values_basis = None
        self.y_values_basis = None

        # data in for fit algorithm
        self.z_values_in = None
        self.x_values, self.y_values = None, None

    def init(self, fit_area, im):
        try:
            # create basis coordinates for later use
            edge_x = im.edge[1]
            edge_y = im.edge[0]
                # TO DO: double check if there is some better syntax
            self.x_values_basis = np.repeat(np.array([range(0, edge_x)]), edge_y, axis=0)
            y_list = []
            for i in range(0, edge_y):
                y_list.append([i] * edge_x)
            self.y_values_basis = np.array(y_list)

            # first fit analysis with founded guess and boundaries
            self.z_values_in = fit_area.data.flatten()
            self.x_values, self.y_values = self.build_xy_values(fit_area)
            self.initial_fit(self.x_values, self.y_values, self.z_values_in)
        except:
            error_message = 'initialize gaussian fit failed'
            self.send_error(error_message)
            self.initial_params = None
            self.result = None
            self.x_values_basis = None
            self.y_values_basis = None
            self.z_values_in = None
            self.x_values, self.y_values = None, None

    def initial_fit(self, x, y, z):
        # first fit analysis with founded guess and boundaries
        try:
            initial_params = self.guess(x, y, z)
            bounds = ([0, 0, 0, 0, 0, -pi / 2, 0], [16382, np.inf, np.inf, np.inf, np.inf, pi / 2, 16382])
            popt, pcov = opt.curve_fit(self.gaussian_model, (x, y), z, p0=initial_params, bounds=bounds)
            [amplitude, centerx, centery, sigmax, sigmay, rot, offset] = popt.tolist()
            self.initial_params = amplitude, centerx, centery, sigmax, sigmay, rot, offset
            self.result = popt
        except:
            error_message = 'initial fit unsuccessful'
            self.send_error(error_message)
            self.initial_params = None
            self.result = None

    def build_xy_values(self, fit_area):
        # slice basis x_y values so their frame fits the fit data
        edge_x = fit_area.edge[1]
        edge_y = fit_area.edge[0]
        slice_x_y = (slice(0, edge_y, None), slice(0, edge_x, None))
        x_values = self.x_values_basis[slice_x_y].flatten()
        y_values = self.y_values_basis[slice_x_y].flatten()
        return x_values, y_values

    def gaussian_model(self, x_y, amplitude, x_center, y_center, sigma_x, sigma_y, theta, offset):
        # calculate test data during fit algorithm to compare with data in
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
        # guess the approximate values for the first fit analysis
        try:
            maxx, minx = np.amax(x), np.amin(x)
            maxy, miny = np.amax(y), np.amin(y)
            maxz, minz = np.amax(z), np.amin(z)

            centerx = x[np.argmax(z)]
            centery = y[np.argmax(z)]
            amplitude = (maxz - minz)  # more or less height

            sigmax = (maxx - minx) / 6.0 # works if frame is close around object
            sigmay = (maxy - miny) / 6.0 # works if frame is close around object
            offset = minz

            return amplitude, centerx, centery, sigmax, sigmay, 0, offset
        except:
            error_message = 'guessing initial fit params failed'
            self.send_error(error_message)
            return 0, 0, 0, 0, 0, 0, 0

    def update(self, fit_area):
        try:
            # get data in
            self.z_values_in = fit_area.data.flatten()
            self.x_values, self.y_values = self.build_xy_values(fit_area)

            # sample values if sampeled is active
            if self.sampled > 1:
                self.z_values_in = self.z_values_in[0:(len(self.z_values_in)):self.sampled]
                self.x_values = self.x_values[0:(len(self.x_values)):self.sampled]
                self.y_values = self.y_values[0:(len(self.y_values)):self.sampled]

            # do actual fit analysis
            self.initial_params = self.result # old result = new initial params
            popt, pcov = opt.curve_fit(self.gaussian_model, (self.x_values, self.y_values), self.z_values_in, p0=self.initial_params)
            self.result = popt
        except:
            error_message = 'updating gaussian fit failed'
            self.send_error(error_message)

    def get_fit_params(self, fit_area):
        # get current fit params
        try:
            [amplitude, centerx, centery, sigmax, sigmay, rot, offset] = self.result.tolist()

            # centerx and centery are so far in the fit_area coordinates and have to be projected to image
            return [amplitude, fit_area.x_coordinate_in_image(centerx), fit_area.y_coordinate_in_image(centery),
                    sigmax, sigmay, rot, offset]
        except:
            error_message = 'obtaining fit parameter failed'
            self.send_error(error_message)
            return None

    def change_by_user(self, control_param_name, value):
        if control_param_name == 'sampled':
            self.sampled = value
            print("set sampled param to", value)
        else:
            error_message = 'changing gaussian fit control parameter failed'
            self.send_error(error_message)

    def send_error(self,error_message):
        print(error_message)
        self.cam_dat_eps.set_da_error(error_message)

    def show(self):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        # old data
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_title("data in")
        ax.scatter(self.x_values, self.y_values, self.z_values_in, c=self.z_values_in, cmap='viridis', linewidth=0.5)

        # fitted new data
        print(self.result)
        z_values_new = self.gaussian_model((self.x_values, self.y_values), *(self.result))
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.set_title("data fitted")
        ax.scatter(self.x_values, self.y_values, z_values_new, c=z_values_new, cmap='viridis', linewidth=0.5)

        plt.show()


class DataAnalyzer:
    '''
    --- data analyzer ---
    contains all areas to reduce fit data and the fit algorithm
    '''

### --- Init
    def __init__(self, cam_dat_eps, init_dict=None):
        self.cam_dat_eps = cam_dat_eps

        try:
            # load control params from init_dict
            # save so they can be used for initial values in epics
            self.control_params = self.load_control_params(init_dict)

            # initialize all areas
            self.image = Image(self.cam_dat_eps)

            self.add_roi_control_params_from_image(self.image)
            self.roi = Roi(self.control_params['roi_x_start'], self.control_params['roi_x_stop'],
                           self.control_params['roi_y_start'], self.control_params['roi_y_stop'], self.cam_dat_eps)

            self.add_fit_area_control_params_from_roi(self.roi)
            self.fit_area = FitArea(self.control_params['factor'], self.control_params['threshold'],
                                    self.cam_dat_eps, self.control_params['median_flt'])

            # initialize gaussian fit
            self.g_fit = GaussianFit(self.control_params['sampled'], self.cam_dat_eps)
        except:
            error_message = 'initializing data analyzer objects failed, maybe change control parameter in init file'
            self.send_error(error_message)

        # check if full initialization is possible
        self.is_init = self.init()
        if self.is_init:
            self.params = self.g_fit.get_fit_params(self.fit_area)
        else:
            self.params = None

    def load_control_params(self, init_dict):
        # add default control params to control param dictionary
        # 'roi_x_stop', 'roi_y_stop', 'threshold' are added later
        default_control_param_values = \
                {'roi_x_start': 0,
                'roi_y_start': 0,
                'factor': 0.1,
                'median_flt': True,
                'sampled': 1}

        control_param_values = default_control_param_values

        if init_dict:
            if 'control_params_values' in init_dict:
                for param, value in init_dict['control_params_values'].items():
                    control_param_values[param] = value

        return control_param_values

    def add_roi_control_params_from_image(self, image):
        # in case there is already a loaded image, the roi is by default as big as the image
        if 'roi_x_stop' not in self.control_params:
            if image.edge is not None:
                self.control_params['roi_x_stop'] = image.edge[1]
            else:
                # worst case: no user specification, no image to decide on a default
                self.control_params['roi_x_stop'] = 0
        if 'roi_y_stop' not in self.control_params:
            if image.edge is not None:
                self.control_params['roi_y_stop'] = image.edge[0]
            else: self.control_params['roi_y_stop'] = 0

    def add_fit_area_control_params_from_roi(self, roi):
        # in case there is already a loaded roi, the threshold can be calculated from it
        if 'threshold' not in self.control_params:
            if roi.data is not None:
                self.control_params['roi_y_stop'] = self.define_threshold(roi.data)

    def define_threshold(self, data):
        # calculated threshold of a for- and background of some data
            # TO DO: check for better or different ways to find the threshold
        try:
            classif = GaussianMixture(n_components=1)
            classif.fit(data.reshape((data.size, 1)))
            threshold = np.mean(classif.means_)
            return threshold
        except:
            error_message = 'threshold calculation unsuccessful'
            self.send_error(error_message)
            return 0

    def init(self):
        success = False
        # check if the data was loaded successfully everywhere
        if self.image.data is not None:
            self.roi.init(self.image)
            if self.roi.data is not None:
                self.fit_area.init(self.roi)
                if self.fit_area.data is not None:
                    self.g_fit.init(self.fit_area, self.image)
                    self.params = self.g_fit.get_fit_params(self.fit_area)
                    if self.params is not None:
                        success = True
        if not success:
            error_message = 'full data analyzer initialization failed'
            self.send_error(error_message)
        return success

### ---  Connection to CamDatEps
    def get_init_control_params(self):
        # for initializing epics control params as well
        return self.control_params

    def get_fit_params(self):
        # check if params have been calculated yet
        if self.params is not None:
            return self.params
        else:
            return [0, 0, 0, 0, 0, 0, 0]

    def get_data_a_settings(self):
        # for saving
        return{'control_params_values': self.control_params}

    def change_by_user(self, area, control_param_name, value):
        if area == 'roi':
            self.roi.change_by_user(control_param_name, value)
        elif area == 'fit_area':
            self.fit_area.change_by_user(control_param_name, value)
        elif area == 'g_fit':
            self.fit_area.change_by_user(control_param_name, value)
        else:
            error_message = 'changing fit data control parameter failed'
            self.send_error(error_message)
            return

        # update for later use
        self.control_params[control_param_name] = value

    def analyze(self):
        # main analyze function
        if self.is_init:
            try:
                self.image.update()
                self.roi.update(self.image)
                self.fit_area.update(self.roi)
                self.g_fit.update(self.fit_area)
                self.params = self.g_fit.get_fit_params(self.fit_area)
                print("successful analyzation")
                self.send_error(None)
            except:
                error_message = 'analyzing failed'
                self.send_error(error_message)
            # error None
        else:
            self.image.update()
            self.is_init = self.init()

    def send_error(self,error_message):
        print(error_message)
        self.cam_dat_eps.set_da_error(error_message)

    def show(self):
        print("image")
        self.image.show()
        print("roi")
        self.roi.show()
        print("fit_area")
        self.fit_area.show()
        print("model")
        self.g_fit.show()



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
                    {'roi_x_start': 500,
                    'roi_x_stop': 1250,
                    'roi_y_start': 600,
                    'roi_y_stop': 1000,
                    'factor': 0.1,
                    'threshold': 1400,
                    'median_flt': True,
                    'sampled': 1}}

    class fake_CamDatEps:
        def __init__(self):
            self.ia = ImageAquirerFile(self, 'D:\\HZB\\Camera_Data\\mls13\\', 200)
            #self.ia = ImageAquirerVimba(fake_CamDatEps, {"features": {"ExposureAuto":"Off"}})
            self.data_analyzer = DataAnalyzer(self, example_init)
            #self.data_analyzer.show()

            for i in range(0, 2):
                self.data_analyzer.analyze()

            self.ia.aquire_sync()
            self.data_analyzer.analyze()
            self.data_analyzer.show()

            for i in range(0, 10):
                self.ia.aquire_sync()
                #self.ia.get_frame()
                self.data_analyzer.analyze()

        def get_image(self):
            return self.ia.get_image()

        def set_da_error(self, error_message):
            pass

    f_camdatep = fake_CamDatEps()