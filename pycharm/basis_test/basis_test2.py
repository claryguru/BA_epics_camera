# Import the basic framework components.
from softioc import softioc, builder, asyncio_dispatcher
import asyncio

###################################################DO NOT USE
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

    def run(self):
            self.ai.set(self.data_a.params[0])
            print("ai set to", self.data_a.params[0])

class DataAnalyzer:
    def __init__(self, ia):
        self.ia = ia
        self.params = [0,0]

    def analyze(self):
        self.params[0] = self.ia.index
        self.params[1] += self.ia.index
        print("params berechnet ", self.params)

class ImageA:
    def __init__(self):
        self.index= 1

    def aquire(self):
        self.index += 1
        print(self.index, " aquired")

# this one is out, weil ia.aquire in sich eine for schleife braucht und nicht jedes mal neu aufgerufen werden kann
# falls epics je Sachen emppängt müssten diese auch unabhängigi vom Rest ausgeführt werden
async def main(ia,data_a,epics):
    while True:
        ia.aquire()
        #####warum nicht die zwei zusammen?
        data_a.analyze()
        epics.run()
        await asyncio.sleep(1)

if __name__ == '__main__':
    # Create an asyncio dispatcher, the event loop is now running
    dispatcher = asyncio_dispatcher.AsyncioDispatcher()

    ia = ImageA()
    data_a = DataAnalyzer(ia)
    epics = Epics(data_a)

    softioc.iocInit(dispatcher)

    # Start processes required to be run after iocInit
    asyncio.run_coroutine_threadsafe(main(ia,data_a,epics), dispatcher.loop)

    # Finally leave the IOC running with an interactive shell.
    softioc.interactive_ioc(globals())