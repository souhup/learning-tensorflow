"""
图形界面。
"""

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from PIL import Image
import numpy as np


class PanelWidget(QWidget):
    """
    手写板界面类。

    """

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.path = QPainterPath()
        self.resize(280, 280)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def paintEvent(self, event):
        painter = QPainter(self)
        pen = QPen()
        pen.setWidth(25)
        painter.setPen(pen)
        painter.drawPath(self.path)

    def mousePressEvent(self, event):
        self.path.moveTo(event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        self.path.lineTo(event.pos())
        self.update()


class Widget(QWidget):
    """
    主界面类。
    """

    def __init__(self, recognize):
        QWidget.__init__(self)
        self.recognize = recognize
        self.setupUi()
        self.show()

    def setupUi(self):
        """
        初始化UI。
        """
        self.setWindowTitle("手写识别")
        self.resize(410, 320)

        self.panelWidget = PanelWidget(self)
        self.panelWidget.setGeometry(QRect(20, 20, 280, 280))

        self.verticalLayoutWidget = QWidget(self)
        self.verticalLayoutWidget.setGeometry(QRect(320, 20, 70, 100))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(1, 0, 0, 0)

        self.clearBtn = QPushButton('清空', self.verticalLayoutWidget)
        self.recBtn = QPushButton('识别', self.verticalLayoutWidget)
        self.exitBtn = QPushButton('退出', self.verticalLayoutWidget)

        self.clearBtn.clicked.connect(self.clearPanel)
        self.recBtn.clicked.connect(self.recPanel)
        self.exitBtn.clicked.connect(self.close)

        self.verticalLayout.addWidget(self.recBtn)
        self.verticalLayout.addWidget(self.clearBtn)
        self.verticalLayout.addWidget(self.exitBtn)

    def keyPressEvent(self, event):
        """
        处理按键。
        """
        if event.key() == Qt.Key_Escape:
            self.close()

    def clearPanel(self):
        """
        清空手写板。
        """
        self.panelWidget.path = QPainterPath()
        self.panelWidget.update()

    def recPanel(self):
        """
        识别手写板内容。
        """
        pixmap = QPixmap(self.panelWidget.size())
        self.panelWidget.render(pixmap)
        pixmap.save('save/digit.png')
        data = self.getPicData()
        result = self.recognize.predict(data)

        messageBox = QMessageBox(self)
        messageBox.setWindowTitle("结果")
        messageBox.setText("数字是："+str(result))
        messageBox.show()

    def getPicData(self):
        """
        得到手写数字的数据。
        """
        im = Image.open('save/digit.png').convert("L").resize((28, 28), Image.ANTIALIAS)
        data = 1 - np.array(im) / 255.0
        im.close()
        return [data.reshape(-1)]
