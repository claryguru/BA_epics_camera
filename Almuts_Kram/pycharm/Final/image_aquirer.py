import asyncio
from vimba import *
import numpy as np
from scipy.ndimage import rotate

class ImageAcquirer:
    '''
    --- image aquirer ---
    empty class, to be filled with from inheriting classes, so it can take their form and is exchangeable
    '''
    def get_ia_settings(self):
        pass

    def get_image(self):
        pass

    def send_error(self, error_message):
        pass

    def get_current_feature(self, feature_name):
        pass

    def set_feature(self, feature_name, value):
        pass

    def stop_running(self):
        pass

    async def aquire(self):
        pass


class ImageAcquirerFile(ImageAcquirer):
    '''
    --- image aquirer based on files ---
    image aquirer for testing purpose
    loades data from filepath
    filesnames are named index+'test.npy'
    max_index: number of test files in testpath
    rot: for testing rotation, test files can be rotated by a with rot specified angle in degrees
    '''
    def __init__(self, cam_dat_eps, file_path, max_index, init_dict=None, rot=0):
        self.index = 1
        self.file_path = file_path
        self.rotation = rot
        self.image = None # self.load_im() -> testing what happens if first image wasn't loaded
        self.max_index = max_index

        self.running = False

        self.init_dict = None
        if init_dict:
            print("init cam from dict")
            self.init_dict = init_dict

        self.cam_dat_eps = cam_dat_eps

    def get_image(self):
        return self.image

    def get_ia_settings(self):
        if self.init_dict:
            return self.init_dict
        else:
            return {}

    def get_current_feature(self, feature_name):
        return 'file_ia_feature'

    def send_error(self, error_message):
        self.cam_dat_eps.set_ia_error(error_message)

    def load_im(self):
        filename = self.file_path + str(self.index) + 'test.npy'
        with open(filename, 'rb') as f:
            ar = np.load(f)
        print("loaded image ", self.index)
        self.index += 1
        ar = ar.reshape(1456, 1936)
        #test rotation
        ar = rotate(ar, self.rotation)
        return ar

    def aquire_sync(self):
        if self.index <= self.max_index:
            self.image = self.load_im()
        else:
            print("reached end of folder")

    def stop_running(self):
        self.running = False

    def set_feature(self, feature_name, value): pass

    async def aquire(self):
        self.running = True
        while True:
            if self.running:
                while (self.index <= self.max_index) and self.running:
                    self.image = self.load_im()
                    await asyncio.sleep(2)
                print("reached end of folder")
                self.index = 0
                await asyncio.sleep(0)
            await asyncio.sleep(0)


class ImageAcquirerVimba(ImageAcquirer):
    '''
    --- image aquirer based on vimba python API ---
    gets data as frame from camera
    to build a connection the camera has to be found and set up each time
    '''
    def __init__(self, cam_dat_eps, init_dict=None):
        self.image = None
        self.cam_id = ""
        self.feature_dict = {}

        self.running = False

        self.cam_dat_eps = cam_dat_eps

        # get cam information from init_dict
        if init_dict:
            if 'cam_id' in init_dict:
                self.cam_id = init_dict['cam_id']
            if 'features' in init_dict:
                self.feature_dict = init_dict['features']

        #load first image with right settings
        self.get_frame()

    ### CamDatEps connection
    def get_image(self):
        # get image for data analyzer
        return self.image

    def get_ia_settings(self):
        # get settings for saving (only get cam settings that were explicitly change by user)
        settings = {}
        if self.cam_id:
            settings['cam_id'] = self.cam_id
        if self.feature_dict:
            settings['features'] = self.feature_dict
        return settings

    def set_feature(self, feature_name, value):
        try:
            self.stop_running()
            # add or change feature
            self.feature_dict[feature_name] = value
                # TO DO: check if connection is actually lost
            # when start running again set up camera with new feature
            self.start_running()
            print(feature_name,'was set to', value)
            self.send_error(None)
        except:
            error_message = 'setting' + feature_name + 'unsuccessful'
            self.send_error(error_message)


    def get_current_feature(self, feature_name):
        value = None
        try:
            with Vimba.get_instance():
                with self.get_camera() as cam:
                    print("Camera has been opened")
                    feat = cam.get_feature_by_name(feature_name)
                    value = feat.get()
                    self.send_error(None)
        except:
            error_message = 'no value for '+feature_name+' found'
            self.send_error(error_message)
        finally:
            return value

    def send_error(self, error_message):
        self.cam_dat_eps.set_ia_error(error_message)

    ### Connect to camera
    def get_camera(self):
        with Vimba.get_instance() as vimba:
            if self.cam_id:
                try:
                    return vimba.get_camera_by_id(self.cam_id)
                except VimbaCameraError:
                    error_message = 'Failed to access Camera '+self.cam_id
                    self.send_error(error_message)
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    error_message = 'No Cameras accessible'
                    self.send_error(error_message)
                return cams[0]

    def set_up(self, cam):
        try:
            for feature, value in self.feature_dict.items():
                feat = cam.get_feature_by_name(feature)
                feat.set(value)
                print(feat, "set to", value)
        except:
            error_message = 'not all features were set'
            self.send_error(error_message)

    def get_frame(self):
        try:
            with Vimba.get_instance():
                with self.get_camera() as cam:
                    print("Camera has been opened")

                    self.set_up(cam)
                    self.image = cam.get_frame().as_numpy_ndarray()
                    print("one image loaded")
                    self.send_error(None)
        except:
            error_message = 'loading one frame failed'
            self.send_error(error_message)

    ### Continuous image acquisition
    def stop_running(self):
        self.running = False

    def start_running(self):
        self.running = True

    async def aquire(self):
        self.running = True
        while True:
            while self.running:
                # while running always try to reconnect
                try:
                    with Vimba.get_instance():
                        with self.get_camera() as cam:
                            print("Camera has been opened")

                            self.set_up(cam)
                            print("Camera set up")

                            # connection was established
                            cam_connection = True
                            error_message = 'camera connected'
                            self.send_error(error_message)
                            while cam_connection and self.running:
                                try:
                                    self.image = cam.get_frame().as_numpy_ndarray()
                                    # print("new image acquired")
                                    await asyncio.sleep(0)
                                except:
                                    error_message = 'Camera problem dedected,trying to reconnect'
                                    self.send_error(error_message)
                                    cam_connection = False
                                    await asyncio.sleep(0)
                except: await asyncio.sleep(0)
            await asyncio.sleep(0)


if __name__ == '__main__':
    #trying out ImageAcquirerFile
    class fake_CamDatEps:
        def __init__(self):
            ia = ImageAcquirerFile(self, 'D:\\HZB\\Camera_Data\\mls13\\', 200, 0)

            init_dict_example = \
                {"features": {"ExposureAuto": "Off"},
                 "cam_id": "DEV_000F31024A32"}

            # trying ImageAcquirerVimba
            ia_cam = ImageAcquirerVimba(self, init_dict_example)

        def get_error(self):
            return 0

        def set_ia_error(self, error_message):
            print(error_message)

    fake_camdateps= fake_CamDatEps()


