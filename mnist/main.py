"""
程序入口。
"""
import sys

from PyQt5.QtWidgets import *

from mnist.cnn_module import Rec
from mnist.user_interface import Widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    rec = Rec()
    w = Widget(rec)
    sys.exit(app.exec_())
