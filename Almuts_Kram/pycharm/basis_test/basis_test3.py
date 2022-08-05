# Import the basic framework components.
from softioc import softioc, builder, asyncio_dispatcher
import asyncio
from vimba import *

class Epics:
    def __init__(self, data_a):
        builder.SetDeviceName("MY-DEVICE-PREFIX")

        # Create some records
        self.ai = builder.aIn('AI', initial_value=5)
        self.ao = builder.aOut('AO', initial_value=12.45, always_update=True,
                          on_update=lambda v: self.ai.set(v))

        # Boilerplate get the IOC started
        builder.LoadDatabase()
        self.data_a = data_a

    async def run(self):
        while True:
            self.ai.set(self.data_a.params[0])
            print("ai set to", self.data_a.params[0])
            await asyncio.sleep(0)

class DataAnalyzer:
    def __init__(self, ia):
        self.ia = ia
        self.params = [0,0]

    async def analyze(self):
        while True:
            self.params[0] = self.ia.index
            self.params[1] += self.ia.index
            print("params berechnet ", self.params)
            await asyncio.sleep(2)

class ImageA:
    def __init__(self):
        self.index= 1

    async def aquire(self):
        print("hi")
        with Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            print(cams)
            if len(cams) > 0:
                with cams[0] as cam:
                    cam.ExposureAuto = 'On'
                    while True:
                        frame = cam.get_frame()
                        self.index += 1
                        print('Got {} {}'.format(frame, self.index), flush=True)
                        await asyncio.sleep(0)
            else:
                print("no camera detected")
                await asyncio.sleep(0)



if __name__ == '__main__':
    # Create an asyncio dispatcher, the event loop is now running
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    ia = ImageA()
    data_a = DataAnalyzer(ia)
    epics = Epics(data_a)

    softioc.iocInit(dispatcher)

    # Start processes required to be run after iocInit
    asyncio.run_coroutine_threadsafe(ia.aquire(), dispatcher.loop)
    asyncio.run_coroutine_threadsafe(data_a.analyze(), dispatcher.loop)
    asyncio.run_coroutine_threadsafe(epics.run(), dispatcher.loop)


    # Finally leave the IOC running with an interactive shell.
    softioc.interactive_ioc(globals())