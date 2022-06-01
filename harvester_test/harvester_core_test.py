from harvesters.core import Harvester
import numpy as np

if __name__ == "__main__":
    h = Harvester()

    #thats the one working:
    h.add_file('C:\\Program Files\\Allied Vision\\Vimba_6.0\\VimbaGigETL\\Bin\\Win64\\VimbaGigETL.cti')
    h.update()
    print(h.device_info_list)

    ia = h.create() #ia: ImageAquirer Object
    ia.start()

    print(ia.remote_device.node_map.DeviceTemperature.value)
    print(ia.remote_device.node_map.Width.value)

    with ia.fetch() as buffer: #ImageAcquirer.fetch() without a time-out value means the function call waits until a buffer is filled up with an image
        component = buffer.payload.components[0]
        oneD = component.data
        print('1D: {0}'.format(oneD))
       # Reshape the NumPy array into a 2D array:
        twoD = component.data.reshape(component.height, component.width)
        print('2D: {0}'.format(twoD))

    ia.stop()
    ia.destroy()
    h.reset()
