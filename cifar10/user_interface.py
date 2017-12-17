"""
图形用户界面
"""
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from cifar10.cnn_module import Rec


class Widget(QWidget):
    """
    识别图片窗口
    """
    def __init__(self):
        QWidget.__init__(self)
        self.rec = Rec()
        self.rec.prepare_detect()
        self.init_ui()
        self.show()

    def init_ui(self):
        """
        绘制界面
        """
        self.setWindowTitle("物体识别")
        self.resize(460, 360)

        self.label = QLabel(self)
        self.label.setGeometry(QRect(20, 20, 320, 320))

        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setGeometry(QRect(360, 20, 80, 100))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        openBtn = QPushButton('打开', self.verticalLayoutWidget)
        exitBtn = QPushButton('退出', self.verticalLayoutWidget)

        openBtn.clicked.connect(self.open_image)
        exitBtn.clicked.connect(self.close)

        self.verticalLayout.addWidget(openBtn)
        self.verticalLayout.addWidget(exitBtn)

    def open_image(self):
        """
        打开并识别图片
        """
        path, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg *.png *.jpeg *.bmp")
        img = QPixmap(path).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(img)
        result = self.rec.detect_image(path)

        messageBox = QMessageBox(self)
        messageBox.setWindowTitle("结果")
        messageBox.setText(result)
        messageBox.show()
