# Import the basic framework components.
from softioc import softioc, asyncio_dispatcher
import asyncio

# Import own classes
from epics import Epics
from data_analyzer import DataAnalyzer
from image_aquirer import ImageAquirerFile, ImageAquirerVimba

# Import others
import json


class CamDatEps:
    def __init__(self, init_dict=None):
        ia_init, data_a_init, epics_init = self.load_cam_dat_ep(init_dict)

        # Create own objects from Classes
            # choose way of image aquiring:
        self.ia = ImageAquirerFile('D:\\HZB\\Camera_Data\\mls13\\', 200, ia_init)
        #self.ia = ImageAquirerVimba(ia_init)

        self.data_a = DataAnalyzer(self, data_a_init)
        self.epics = Epics(self, self.data_a.get_init_control_params(), epics_init)

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
        settings = {'camera': self.ia.get_ia_settings(),
                    'data_a': self.data_a.get_data_a_settings(),
                    'epics': self.epics.get_epics_settings()}
        return settings

    def on_update(self, area, ctr_param_name, value):
        self.data_a.change_by_user(area, ctr_param_name, value)

    def analyze_and_run(self):
        while True:
            self.data_a.analyze()
            self.epics.run()

    def run(self, dispatcher):
        asyncio.run_coroutine_threadsafe(self.ia.aquire(), dispatcher.loop)
        asyncio.run_coroutine_threadsafe(self.data_a.analyze(), dispatcher.loop)
        asyncio.run_coroutine_threadsafe(self.epics.run(), dispatcher.loop)


class CamIOC:
    def __init__(self):
        # Create an asyncio dispatcher, the event loop is now running
        self.dispatcher = asyncio_dispatcher.AsyncioDispatcher()

        # Create a List of used CamDatEps:
        self.cams = []

    def add_camera(self, init_file=None):
        #falls init file lade set_up dict und init Camera übergeben
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
        # Start iocInit
        softioc.iocInit(self.dispatcher)

        # Start all Cameras, their data analysation and ioc connection
        for cam in self.cams:
            cam.run(dispatcher=self.dispatcher)

        # Finally leave the IOC running with an interactive shell.
        # might not be necessary:
        softioc.interactive_ioc(globals())



if __name__ == '__main__':
    #list_ camera Name, device_name, Cam Einstellungen -> Ines
    ioc = CamIOC()
    ioc.add_camera('.\\init_files\\init_example.json')
    ioc.save_toJson('.\\init_files')
    ioc.run()



