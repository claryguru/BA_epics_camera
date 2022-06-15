import sys
from PyQt5.QtWidgets import QApplication
from harvesters_gui.frontend.pyqt5 import Harvester
#change python interpreter zu 3.7
# choose VimbaGigETL.cti as driver

if __name__ == '__main__':
    app = QApplication(sys.argv)
    h = Harvester()
    h.show()
    sys.exit(app.exec_())