'''
    功能：
        利用训练好的模型，进行实时手写体识别
'''

import tensorflow.compat.v1 as tf
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PIL import ImageGrab, Image

# 采用tensorflow1.x版本
tf.compat.v1.disable_eager_execution()
Window_width = 284
Window_height = 330


class MyMnistWindow(QWidget):

    def __init__(self):
        super(MyMnistWindow, self).__init__()

        self.resize(Window_width, Window_height)  # resize设置宽高
        self.move(100, 100)    # move设置位置
        self.setWindowFlags(Qt.FramelessWindowHint)  # 窗体无边框
        # setMouseTracking设置为False，否则不按下鼠标时也会跟踪鼠标事件
        self.setMouseTracking(False)

        self.window_x = 0
        self.window_y = 0  # 记录窗口的左上顶点
        self.pos_xy = []  # 保存鼠标移动过的点

        # 添加一系列控件
        self.label_draw = QLabel('', self)
        self.label_draw.setGeometry(2, 2, 280, 280)
        self.label_draw.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        self.label_result_name = QLabel('识别结果：', self)
        self.label_result_name.setGeometry(2, 290, 60, 35)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        self.label_result = QLabel(' ', self)
        self.label_result.setGeometry(64, 290, 35, 35)
        self.label_result.setFont(QFont("Roman times", 8, QFont.Bold))
        self.label_result.setStyleSheet("QLabel{border:1px solid black;}")
        self.label_result.setAlignment(Qt.AlignCenter)

        self.btn_recognize = QPushButton("识别", self)
        self.btn_recognize.setGeometry(110, 290, 50, 35)
        self.btn_recognize.clicked.connect(self.btn_recognize_on_clicked)

        self.btn_clear = QPushButton("清空", self)
        self.btn_clear.setGeometry(170, 290, 50, 35)
        self.btn_clear.clicked.connect(self.btn_clear_on_clicked)

        self.btn_close = QPushButton("输出", self)
        self.btn_close.setGeometry(230, 290, 50, 35)
        self.btn_close.clicked.connect(self.btn_close_on_clicked)

        self.result = 0

    def move_to_xy(self, x, y):
        self.window_x = x
        self.window_y = y
        self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        pen = QPen(Qt.black, 30, Qt.SolidLine)
        painter.setPen(pen)
        '''
            首先判断pos_xy列表中是不是至少有两个点了
            然后将pos_xy中第一个点赋值给point_start
            利用中间变量pos_tmp遍历整个pos_xy列表
                point_end = pos_tmp

                判断point_end是否是断点，如果是
                    point_start赋值为断点
                    continue
                判断point_start是否是断点，如果是
                    point_start赋值为point_end
                    continue

                画point_start到point_end之间的线
                point_start = point_end
            这样，不断地将相邻两个点之间画线，就能留下鼠标移动轨迹了
        '''
        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(
                    point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    def mouseMoveEvent(self, event):
        '''
            按住鼠标移动事件：将当前点添加到pos_xy列表中
            调用update()函数在这里相当于调用paintEvent()函数
            每次update()时，之前调用的paintEvent()留下的痕迹都会清空
        '''
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)

        self.update()

    def mouseReleaseEvent(self, event):
        '''
            重写鼠标按住后松开的事件
            在每次松开后向pos_xy列表中添加一个断点(-1, -1)
            然后在绘画时判断一下是不是断点就行了
            是断点的话就跳过去，不与之前的连续
        '''
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

        self.update()

    def btn_recognize_on_clicked(self):
        bbox = (self.window_x + 4, self.window_y + 4, self.window_x +
                Window_width - 4, self.window_y + Window_width - 4)
        im = ImageGrab.grab(bbox)    # 截屏，手写数字部分
        im = im.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素

        recognize_result = self.recognize_img(im)  # 调用识别函数

        self.label_result.setText(str(recognize_result))  # 显示识别结果
        self.result = recognize_result
        self.update()

    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.label_result.setText('')
        self.update()

    def btn_close_on_clicked(self):
        self.close()

    def recognize_img(self, img):
        myimage = img.convert('L')  # 转换成灰度图
        tv = list(myimage.getdata())  # 获取图片像素值
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]  # 转换像素范围到[0 1], 0是纯白 1是纯黑

        init = tf.global_variables_initializer()
        saver = tf.train.Saver

        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.import_meta_graph(
                './HandWriting/minst_cnn_model.ckpt.meta')  # 载入模型结构
            saver.restore(sess, './HandWriting/minst_cnn_model.ckpt')  # 载入模型参数

            graph = tf.get_default_graph()  # 加载计算图
            x = graph.get_tensor_by_name("x:0")  # 从模型中读取占位符变量
            keep_prob = graph.get_tensor_by_name("keep_prob:0")
            y_conv = graph.get_tensor_by_name("y_conv:0")  # 关键的一句  从模型中读取占位符变量

            prediction = tf.argmax(y_conv, 1)
            # feed_dict输入数据给placeholder占位符
            predint = prediction.eval(
                feed_dict={x: [tva], keep_prob: 1.0}, session=sess)
            #print(predint[0])
        return predint[0]


def handwriting_result():
    app = QApplication(sys.argv)
    # 四个手写识别框依次识别
    mymnist_1 = MyMnistWindow()
    mymnist_1.move_to_xy(100, 100)
    mymnist_1.show()
    mymnist_2 = MyMnistWindow()
    mymnist_2.move_to_xy(100+Window_width, 100)
    mymnist_2.show()
    mymnist_3 = MyMnistWindow()
    mymnist_3.move_to_xy(100+Window_width*2, 100)
    mymnist_3.show()
    mymnist_4 = MyMnistWindow()
    mymnist_4.move_to_xy(100+Window_width*3, 100)
    mymnist_4.show()
    app.exec_()
    x_coor = mymnist_1.result*10+mymnist_2.result  # 识别结果运算成坐标
    y_coor = mymnist_3.result*10+mymnist_4.result
    return x_coor, y_coor
