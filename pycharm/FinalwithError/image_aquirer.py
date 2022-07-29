import asyncio
from vimba import *
import numpy as np
from scipy.ndimage import rotate

class ImageAquirer:
    def get_ia_settings(self):
        pass

    def get_image(self):
        pass

    def notify_error(self):
        pass

    def get_current_feature(self, feature_name):
        pass

    def set_feature(self, feature_name, value):
        pass

    def stop_running(self):
        pass

    async def aquire(self):
        pass



class ImageAquirerFile(ImageAquirer):
    def __init__(self, cam_dat_eps, file_path, max_index, init_dict=None, rot=0):
        self.index = 1
        self.file_path = file_path
        self.rotation = rot
        self.image = self.load_im()
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

    def notify_error(self, error_message):
        self.cam_dat_eps.notify_ia_error(error_message)

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
        while (self.index <= self.max_index) and self.running:
            self.image = self.load_im()
            await asyncio.sleep(2)
        print("reached end of folder")
        await asyncio.sleep(0)


class ImageAquirerVimba(ImageAquirer):
    def __init__(self, cam_dat_eps, init_dict=None):
        self.image = None
        self.cam_id = ""
        self.feature_dict = {}

        self.running = False

        self.cam_dat_eps = cam_dat_eps

        if init_dict:
            if 'cam_id' in init_dict:
                self.cam_id = init_dict['cam_id']
            if 'features' in init_dict:
                self.feature_dict = init_dict['features']

        #load first image with right settings
        self.get_frame()

    def get_image(self):
        print("get_image", type(self.image))
        return self.image

    def get_ia_settings(self):
        settings = {}
        if self.cam_id:
            settings['cam_id'] = self.cam_id
        if self.feature_dict:
            settings['features'] = self.feature_dict
        return settings

    def get_camera(self):
        with Vimba.get_instance() as vimba:
            if self.cam_id:
                try:
                    return vimba.get_camera_by_id(self.cam_id)
                except VimbaCameraError:
                    self.cam_dat_eps.set_ia_error('Failed to access Camera '+self.cam_id)
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    self.cam_dat_eps.set_ia_error('No Cameras accessible')
                return cams[0]

    def set_feature(self, feature_name, value):
        self.stop_running()
        self.feature_dict[feature_name]=value
        self.start_running()


    def set_up(self, cam):
        for feature, value in self.feature_dict.items():
            feat = cam.get_feature_by_name(feature)
            feat.set(value)
            print("feature set to", feat)

    def get_frame(self):
        try:
            with Vimba.get_instance():
                with self.get_camera() as cam:
                    print("Camera has been opened")

                    self.set_up(cam)
                    self.image = cam.get_frame().as_numpy_ndarray()
                    print("one image loaded")
        except:
            pass #dummy image?

    def get_current_feature(self, feature_name):
        try:
            with Vimba.get_instance():
                with self.get_camera() as cam:
                    print("Camera has been opened")
                    feat = cam.get_feature_by_name(feature_name)
                    value = feat.get()
        except:
            value = 'no value for '+feature_name+' found'
        finally:
            return value

    def stop_running(self):
        self.running = False

    def start_running(self):
        self.running = True

    async def aquire(self):
        self.running = True
        while True:
            while self.running:
                with Vimba.get_instance():
                    with self.get_camera() as cam:
                        print("Camera has been opened")

                        self.set_up(cam)
                        print("Camera set up")

                        cam_connection = True
                        while cam_connection and self.running:
                            try:
                                self.image = cam.get_frame().as_numpy_ndarray()
                                print("new image aquired")
                                await asyncio.sleep(2)
                            except:
                                self.cam_dat_eps.set_ia_error('Camera problem dedected,trying to reconnect')
                                cam_connection = False
                                await asyncio.sleep(2)
            await asyncio.sleep(0)


if __name__ == '__main__':
    #trying out ImageAquirerFile
    class fake_CamDatEps:
        def __init__(self):
            ia = ImageAquirerFile(self, 'D:\\HZB\\Camera_Data\\mls13\\', 200, 0)

            init_dict_example = \
                {"features": {"ExposureAuto": "Off"},
                 "cam_id": "DEV_000F31024A32"}

            # trying ImageAquirerVimba
            ia_cam = ImageAquirerVimba(init_dict_example)

        def get_error(self):
            return 0

    fake_camdateps= fake_CamDatEps()


