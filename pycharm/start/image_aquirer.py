import asyncio
from vimba import *




class ImageAquirer:
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