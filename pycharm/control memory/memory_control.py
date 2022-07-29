import psutil
import time

if __name__=='__main__':
    while True:
        p = psutil.Process(8488)
        print(p.memory_info())
        time.sleep(30)

