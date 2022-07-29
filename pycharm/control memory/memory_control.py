import psutil
import time

if __name__=='__main__':
    p = psutil.Process(5060)
    print(p)
    print(p.memory_info())
    time.sleep(30)
    p = psutil.Process(5060)
    print(p.memory_info())
