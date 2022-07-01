from softioc import builder
import asyncio

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
            print("ai set to 0. Parameter:", self.data_a.params[0])
            await asyncio.sleep(0)

            self

#ao Controlwerte ankommen