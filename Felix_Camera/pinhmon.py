#!/usr/bin/python
# -*- coding: utf-8 -*-
''' pymbavimbaGUI
author:     felix.kramer(at)physik.hu-berlin.de
'''
from __future__ import print_function, division
#from accpy.measure.av import features
from vimba import Vimba
from time import sleep, time
from threading import Thread
try:
    import Tkinter as tk
    from Tkinter import N, E, S, W
    from tkMessageBox import askokcancel
    from Queue import Queue, Empty, Full
except ImportError:
    import tkinter as tk
    from tkinter import N, E, S, W
    from tkinter.messagebox import askokcancel
    from queue import Queue, Empty, Full
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import subplots
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#noise = lambda: np.random.randint(0, 2**14, [1456, 1936], dtype='uint16')
noise = lambda: np.random.randint(0, 2**14, [182, 242], dtype='uint16')


def camcontrol(q, infovars):
    with Vimba() as vimba:
        system = vimba.getSystem()
        system.runFeatureCommand('GeVDiscoveryAllOnce')
        sleep(0.2)

        try:
#            c0 = vimba.getCamera(vimba.getCameraIds()[0])
            c0 = vimba.getCamera('DEV_000F31024A32')
            print("cam found")
        except IndexError:
            return
        c0.openCamera()
        c0.StreamBytesPerSecond = 100000000

#        cnt = 0
#        for feature in features:
#            try:
#                print(feature, getattr(c0, feature))
#            except:
#                print(feature, 'not implemented')
#                cnt += 1
#        print(cnt)

        # create and start camconf and framegrabber threads
        threads = [Thread(target=framegrabber, args=(c0, q))]
        threads.append(Thread(target=camconfigure, args=(c0, q)))
        threads.append(Thread(target=camtemp, args=(c0, q, infovars)))
        for t in threads:
            t.setDaemon(True)
            t.start()

        q['kill_camcontrol'].get()
        c0.runFeatureCommand('AcquisitionStop')
        c0.endCapture()
        c0.revokeAllFrames()
        c0.closeCamera()
    return


def framegrabber(c0, q):
    frame = c0.getFrame()
    frame.announceFrame()
    c0.startCapture()

    c0.runFeatureCommand('AcquisitionStart')
    frame.queueFrameCapture()  #queue the first frame

    while q['run'].empty():
        try:
            frame.waitFrameCapture(1000)
            frame.queueFrameCapture()
            success = True
        except:
            app.root.event_generate('<<grabberdrops>>', when='tail')
            success = False

        frame_data = frame.getBufferByteData()
        if success:
            img = np.ndarray(buffer=frame_data,
                             dtype=np.uint16,
                             shape=(frame.height, frame.width))
            try:
                q['plotter'].put_nowait(img)
            except Full:
                app.root.event_generate('<<plotterdrops>>', when='tail')
            pass
            app.root.event_generate('<<frame_grabbed>>', when='tail')
    return


def camconfigure(c0, q):
    while q['run'].empty():
        conf = q['que_camconfigure'].get()
        if conf == 0:
            continue
        for key, val in conf[0].iteritems():
            setattr(c0, key, val[0])
        for key, val in conf[1].iteritems():
            setattr(c0, key, val[0])
        c0.Height = c0.HeightMax
        c0.Width = c0.WidthMax
    return


def camtemp(c0, q, infovars):
    while q['run'].empty():
        infovars[6].set('Temp:   {:<.1f}°C'.format(c0.DeviceTemperature))
        sleep(.1)
    return


def plotter(q, fig, im):
    while q['run_plotter'].empty():
        img = q['plotter'].get()
        im.set_data(img)
        fig.draw_artist(im)
        fig.canvas.blit()
        app.root.event_generate('<<update_label>>', when='tail')
    return


class gui:
    w, h = 800, 600
    vmin, vmax = 0, 2**14 - 1
    grabbedframes = np.zeros(1, dtype='uint32')
    grabberdrops = np.zeros(1, dtype='uint32')
    plottedframes = np.zeros(1, dtype='uint32')
    plotterdrops = np.zeros(1, dtype='uint32')
    plotframe20 = np.zeros(1, dtype='uint32')
    grabframe20 = np.zeros(1, dtype='uint32')
    defaultarray = noise()

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('pinholemonitor')
        self.root.protocol('WM_DELETE_WINDOW', self.kill)
        self.initialize()
        self.layout()
        self.root.bind('<<update_label>>', self.updatelabels)
        self.root.bind('<<grabberdrops>>', lambda event: self.grabberdrops.__iadd__(1))
        self.root.bind('<<plotterdrops>>', lambda event: self.plotterdrops.__iadd__(1))
        self.root.bind('<<frame_grabbed>>', self.grabbed)
        self.starttime = time()

    def grabbed(self, event):
        self.grabbedframes.__iadd__(1)
        self.grabframe20.__iadd__(1)

    def initialize(self):
        # create queues
        self.q = {}
        # First in, First Out (FIFO) is standard otherwise use LifoQueue
        self.q['run'] = Queue(maxsize=1)
        self.q['kill_camcontrol'] = Queue(maxsize=1)
        self.q['que_camconfigure'] = Queue()
        self.q['run_plotter'] = Queue(maxsize=1)
        self.q['plotter'] = Queue(maxsize=1)
        # create dictionary of cam config
        self.confdict = [{'AcquisitionMode'   : ['Continuous', 'MultiFrame', 'Recorder', 'SingleFrame'],
                          'PixelFormat'       : ['Mono14', 'Mono12', 'Mono8']},
                         {'ExposureTimeAbs'   : [100, [10, 1000000, 1]],
                          'Gain'              : [0, [0, 33, 1]],
                          'BinningHorizontal' : [8, [1, 8, 1]],
                          'BinningVertical'   : [8, [1, 8, 1]],
                          'AcquisitionFrameRateAbs': [10, [1, 60, 1]]}]
        self.q['que_camconfigure'].put(self.confdict)

    def layout(self):
        # PLOT FRAME AND THREAD
        lf_P = tk.LabelFrame(self.root, text='Image', padx=5, pady=5)
        lf_P.grid(row=0, column=0, sticky=N+E+S+W, padx=10, pady=10)
        fig, ax = subplots(1, 1, frameon=False, figsize=[8, 6])
        fig.tight_layout()
        im = ax.imshow(self.defaultarray, vmin=self.vmin, vmax=self.vmax)
        ax.autoscale(False)
        ax.grid(False)
        ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        canvas = FigureCanvasTkAgg(fig, master=lf_P)
        canvas.get_tk_widget().pack()
        t = Thread(target=plotter, args=(self.q, fig, im))
        t.setDaemon(True)
        t.start()
        # CONTROL FRAME
        lf_C = tk.LabelFrame(self.root, text='Control', padx=5, pady=5)
        lf_C.grid(row=1, column=0, sticky=N+E+S+W, padx=10, pady=10)
        self.sb = {}
        for key, val in self.confdict[1].items():
            tk.Label(lf_C, text=key).pack()
            i, f, d = val[1][0], val[1][1], val[1][2]
            self.sb[key] = tk.Spinbox(lf_C, from_=i, to=f, increment=d, command=self.qfiller)
            self.sb[key].delete(0, 'end')
            self.sb[key].insert(0, val[0])
            self.sb[key].bind('<Return>', lambda event:self.qfiller())
            self.sb[key].bind('<Button-4>', lambda event,sb=self.sb[key]:sb.invoke('buttonup'))
            self.sb[key].bind('<Button-5>', lambda event,sb=self.sb[key]:sb.invoke('buttondown'))
            self.sb[key].pack()
        # INFO FRAME
        lf_I = tk.LabelFrame(self.root, text='Info', padx=5, pady=5)
        lf_I.grid(row=0, column=1, sticky=N+E+S+W, padx=10, pady=10)
        self.infovars = [tk.StringVar(value='') for _ in range(7)]
        for infovar in self.infovars:
            tk.Label(lf_I, textvariable=infovar, anchor=W, justify=tk.LEFT, width=23, font='mono').pack()
        # START/STOP FRAME
        lf_S = tk.LabelFrame(self.root, text='Run', padx=5, pady=5)
        lf_S.grid(row=1, column=1, sticky=N+E+S+W, padx=10, pady=10)
        self.startstop_label = tk.StringVar(value='Start')
        self.idlestate_label = tk.StringVar(value='idle...')
        startstop = tk.Button(master=lf_S, textvariable=self.startstop_label, command=self.startstop_callback)
        idlestate = tk.Label(lf_S, textvariable=self.idlestate_label)
        startstop.grid(row=0, column=0)
        idlestate.grid(row=0, column=1)

    def startthreads(self):
        try:
            self.q['run'].get_nowait()
        except:
            pass
        t = Thread(target=camcontrol, args=(self.q, self.infovars))
        t.setDaemon(True)
        t.start()
        t.join(timeout=1)
        return t

    def stopthreads(self):
        try:
            self.q['run'].put_nowait(0)
        except Full:
            pass
        try:
            self.q['que_camconfigure'].put_nowait(0)
        except Full:
            pass
        sleep(.5)
        self.q['kill_camcontrol'].put(1)

    def startstop_callback(self):
        if self.startstop_label.get() == 'Start':
            t = self.startthreads()
            if t.isAlive():
                pass
            else:
                self.idlestate_label.set('No Camera found !')
                return
            self.startstop_label.set('Stop')
            self.idlestate_label.set('running...')
            self.starttime = time()
            self.plotframe20 = np.zeros(1, dtype='uint32')
            self.grabframe20 = np.zeros(1, dtype='uint32')
        else:
            self.stopthreads()
            self.startstop_label.set('Start')
            self.idlestate_label.set('idle...')
        return

    def qfiller(self):
        for key in self.sb:
            self.confdict[1][key][0] = int(self.sb[key].get())
        self.q['que_camconfigure'].put(self.confdict, False)

    def updatelabels(self, event):
        self.plottedframes.__iadd__(1)
        self.plotframe20.__iadd__(1)
        dt = time() - self.starttime
        plotfps = self.plotframe20[0]/dt
        grabfps = self.grabframe20[0]/dt
        self.infovars[0].set('DRAW FPS:       {:<.1f}'.format(plotfps))
        self.infovars[1].set('GRAB FPS:       {:<.1f}'.format(grabfps))
        self.infovars[2].set('GRABBED FRAMES: {:<10}'.format(self.grabbedframes[0]))
        self.infovars[3].set('PLOTTED FRAMES: {:<10}'.format(self.plottedframes[0]))
        self.infovars[4].set('GRABBER DROPS:  {:<10}'.format(self.grabberdrops[0]))
        self.infovars[5].set('PLOTTER DROPS:  {:<10}'.format(self.plotterdrops[0]))
        if dt > 20:
            self.starttime = time()
            self.plotframe20 = np.zeros(1, dtype='uint32')
            self.grabframe20 = np.zeros(1, dtype='uint32')

    def kill(self):
        if askokcancel('Quit', 'Do you want to quit?'):
            if self.startstop_label.get() == 'Start':
                self.q['run_plotter'].put_nowait(0)
                try:
                    self.q['plotter'].put_nowait(self.defaultarray)
                except Full:
                    pass
                self.stopthreads()
            self.root.after(500)
            self.root.destroy()
            self.root.quit()


if __name__ == '__main__':
    app = gui()
    app.root.mainloop()