# Import the basic framework components.
from softioc import softioc, asyncio_dispatcher
import asyncio

# Import own classes
from epics import Epics
from data_analyzer import DataAnalyzer
from image_aquirer import ImageAquirerFile, ImageAquirerVimba

class CamDatEps:
    def __init__(self, init_dic = None):
        # Create own objects from Classes
            # choose way of image aquiring:
        self.ia = ImageAquirerFile('D:\\HZB\\Camera_Data\\mls13\\', 200)
        #self.ia = ImageAquirerVimba()

        self.data_a = DataAnalyzer(self)
        self.epics = Epics(self)

        #falls init dic vorhanden  -> dict unterteile weiter in inits geben 
    def on_update(self, area, ctr_param_name, value):
        self.data_a.change_by_user(area, ctr_param_name, value)

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

    def add_camera(self, init_file = None):
        #falls init file lade set_up dict und init Camera übergeben 
        new_cam = CamDatEps()
        self.cams.append(new_cam)

    def set_up_camera(self):
        pass

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
    ioc.add_camera()
    ioc.run()



