"""
程序入口。
"""
import sys

from PyQt5.QtWidgets import *
from cifar10.user_interface import Widget

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    sys.exit(app.exec_())
