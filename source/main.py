# Import the basic framework components.
from softioc import softioc, asyncio_dispatcher
import asyncio

# Import own classes
from epics import Epics, Builder
from data_analyzer import DataAnalyzer
from image_aquirer import ImageAcquirerFile, ImageAcquirerVimba

# Import others
import json


class CamDatEps:
    '''
    --- camera, data analyzer, epics ---
    bringing ia, data analyzer and epics together for one camera
    ia, data_a, epics attributes are protected, so they are only accessed from CamDatEps functions
    '''
    def __init__(self, init_dict=None):
        ia_init, data_a_init, epics_init = self.load_cam_dat_ep(init_dict)

        self.ia_error = None
        self.da_error = None

        # Create own objects from Classes
            # choose way of image aquiring:
        self.__ia = ImageAcquirerFile(self, 'D:\\HZB\\Camera_Data\\mls13\\', 200, ia_init)
        # self.__ia = ImageAcquirerVimba(self, ia_init)

        self.__data_a = DataAnalyzer(self, data_a_init)
        self.__epics = Epics(self, self.__data_a.get_init_control_params(), epics_init)

    def load_cam_dat_ep(self, init_dict=None):
        ia_init, data_a_init, epics_init = None, None, None
        if init_dict:
            if 'camera' in init_dict:
                ia_init = init_dict['camera']
            if 'data_a' in init_dict:
                data_a_init = init_dict['data_a']
            if 'epics' in init_dict:
                epics_init = init_dict['epics']
        return ia_init, data_a_init, epics_init

    def get_cam_dat_ep_settings(self):
        settings = {'camera': self.__ia.get_ia_settings(),
                    'data_a': self.__data_a.get_data_a_settings(),
                    'epics': self.__epics.get_epics_settings()}
        return settings

    def save_toJson(self, file_name):
            settings = self.get_cam_dat_ep_settings()
            file_path = file_name+ '.json'
            with open(file_path, "w") as fp:
                json.dump(settings, fp, indent=4)
                fp.close()

    ### Connecting ia, data_a, epics
    def set_ia_error(self, error_message):
        self.ia_error = error_message
        # don't change error if there is already a previous one
        if (self.ia_error is None) or (self.ia_error == 'camera connected'):
            self.ia_error = error_message
        # than only delete error, in case it dissolved
        if error_message is None:
            self.ia_error = None

    def set_da_error(self, error_message):
            # TO DO: make sure even small errors that have not lead to a fail of analyzation are made visible
        if self.da_error is None:
            self.da_error = error_message
        if error_message is None:
            self.da_error = None

    def get_errors(self):
        return self.ia_error, self.da_error

    def get_current_ia_feature(self, feature_name):
        return self.__ia.get_current_feature(feature_name)

    def set_ia_feature(self, feature_name, value):
        self.__ia.set_feature(feature_name, value)

    def on_control_params_update(self, area, control_param_name, value):
        self.__data_a.change_by_user(area, control_param_name, value)

    def get_image(self):
        return self.__ia.get_image()

    def get_current_params(self):
        return self.__data_a.get_fit_params()

    async def analyze_and_run(self):
        while True:
            self.__data_a.analyze()
            self.__epics.run()
            await asyncio.sleep(2)

    def run(self, dispatcher):
        asyncio.run_coroutine_threadsafe(self.__ia.aquire(), dispatcher.loop)
        asyncio.run_coroutine_threadsafe(self.analyze_and_run(), dispatcher.loop)


class CamIOC:
    '''
    --- camera IOC ---
    main class, responsible for organizing several cameras
    '''
    def __init__(self, builder_name):
        # Create an asyncio dispatcher, the event loop is now running
        self.dispatcher = asyncio_dispatcher.AsyncioDispatcher()

        # Create a List of used CamDatEps:
        self.cams = []

        # Create builder, so records can be created
        self.builder = Builder(builder_name)

    def add_camera(self, init_file=None):
        # in case there is an init file load it and give as dict to CamDatEps
        if init_file:
            init_dict = self.dict_fromJson(init_file)
            new_cam = CamDatEps(init_dict)
        else:
            new_cam = CamDatEps()
        self.cams.append(new_cam)

    def dict_fromJson(self, file_path):
        with open(file_path) as json_data:
            d = json.load(json_data)
            json_data.close()
        return d

    def save_toJson(self, path_to_folder):
        for index, cam in enumerate(self.cams):
            settings = cam.get_cam_dat_ep_settings()
            file_path = path_to_folder + "\\" + 'cam' + str(index) + '.json'
            with open(file_path, "w") as fp:
                json.dump(settings, fp, indent=4)
                fp.close()

    def run(self):
        #load builder Database
        self.builder.start()

        # Start iocInit
        softioc.iocInit(self.dispatcher)


        # Start all Cameras, their data analysation and ioc connection
        for cam in self.cams:
            cam.run(dispatcher=self.dispatcher)

        # Finally leave the IOC running with an interactive shell.
        # might not be necessary:
        softioc.interactive_ioc(globals())



if __name__ == '__main__':
    ioc = CamIOC('CAM_IOC')
    #ioc.add_camera('.\\init_files\\init_example4.json')
    ioc.add_camera('.\\init_files\\init_example.json')
    ioc.save_toJson('.\\init_files')
    ioc.run()



