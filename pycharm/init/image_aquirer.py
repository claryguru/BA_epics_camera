import asyncio
from vimba import *
import numpy as np

class ImageAquirerInterface:
    def set_up(self):
        pass

    def get_image(self):
        pass

    async def aquire(self):
        pass



class ImageAquirerFile(ImageAquirerInterface):
    def __init__(self, file_path, max_index):
        self.index = 1
        self.file_path = file_path
        self.image = self.load_im()
        self.max_index = max_index

    def set_up(self):
        pass

    def get_image(self):
        return self.image

    def load_im(self):
        filename = self.file_path + str(self.index) + 'test.npy'
        with open(filename, 'rb') as f:
            ar = np.load(f)
        print("loaded image ", self.index)
        self.index += 1
        ar = ar.reshape(1456, 1936)
        return ar

    def aquire_sync(self):
        if self.index <= self.max_index:
            self.image = self.load_im()
        else:
            print("reached end of folder")

    async def aquire(self):
        while self.index <= self.max_index:
            self.image = self.load_im()
            await asyncio.sleep(2)
        print("reached end of folder")
        await asyncio.sleep(0)


class ImageAquirerVimba(ImageAquirerInterface):
    def __init__(self, init_dict = None):
        #
        self.cam_id = None
        self.image = None

    def set_up(self):
        pass

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
    ia = ImageAquirerFile('D:\\HZB\\Camera_Data\\mls13\\', 200)