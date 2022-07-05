# Import the basic framework components.
from softioc import softioc, builder, asyncio_dispatcher
import asyncio
from vimba import *
import numpy as np

class Camera:
    def __init__(self):
        # Create own objects from Classes
            # choose way of image aquiring:
        self.ia = ImageA(self)
        #self.ia = ImageAquirerVimba()

        self.data_a = DataAnalyzer(self)
        self.epics = Epics(self)
        #

    def run(self, dispatcher):
        asyncio.run_coroutine_threadsafe(self.ia.aquire(), dispatcher.loop)
        asyncio.run_coroutine_threadsafe(self.data_a.analyze(), dispatcher.loop)
        asyncio.run_coroutine_threadsafe(self.epics.run(), dispatcher.loop)

class Epics:
    def __init__(self, cam):
        builder.SetDeviceName("ALMUT")

        # Create some records
        self.ai = builder.aIn('AI', initial_value=5)
        self.ao = builder.aOut('AO', initial_value=12.45, always_update=True,
                          on_update_name=lambda v, n: self.on_update(n,v))

        # Boilerplate get the IOC started
        builder.LoadDatabase()
        self.data_a = cam.data_a

    def on_update(self, ao_name, value):
        print(ao_name, " change to ", value)


    async def run(self):
        while True:
            self.ai.set(self.data_a.params[0])
            print("ai set to", self.data_a.params[0])
            await asyncio.sleep(1)

class DataAnalyzer:
    def __init__(self, cam):
        self.ia = cam.ia
        self.params = [0,0]

    async def analyze(self):
        while True:
            self.params[0] = self.ia.index
            self.params[1] += self.ia.index
            print("params berechnet ", self.params)
            await asyncio.sleep(2)

class ImageA():
    def __init__(self, cam):
        self.index = 1

    def load_im(self):
        print("loaded image ", self.index)
        self.index += 1

    async def aquire(self):
        while True:
            self.load_im()
            await asyncio.sleep(1)
        print("reached end of folder")
        await asyncio.sleep(0)



if __name__ == '__main__':
    # Create an asyncio dispatcher, the event loop is now running
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    cam = Camera()

    softioc.iocInit(dispatcher)

    # Start processes required to be run after iocInit
    cam.run(dispatcher)


    # Finally leave the IOC running with an interactive shell.
    softioc.interactive_ioc(globals())