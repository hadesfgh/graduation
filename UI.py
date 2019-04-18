# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from main_encoder import mainencoder
from main_pca import mainpca


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.pushButton_main = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_main.setGeometry(QtCore.QRect(240, 400, 120, 40))
        self.pushButton_main.setObjectName("pushButton_main")

        self.pushButton_exit = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_exit.setGeometry(QtCore.QRect(450, 500, 93, 31))
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.pushButton_pic = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_pic.setGeometry(QtCore.QRect(225, 500, 150, 31))
        self.pushButton_pic.setObjectName("pushButton_pic")

        self.pushButton_introduction = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_introduction.setGeometry(QtCore.QRect(50, 500, 91, 31))
        self.pushButton_introduction.setObjectName("pushButton_introduction")

        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setGeometry(QtCore.QRect(1290, 30, 50, 30))
        self.pushButton_open.setObjectName("pushButton_open")

        self.pushButton_export = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_export.setGeometry(QtCore.QRect(1350, 30, 50, 30))
        self.pushButton_export.setObjectName("pushButton_export")

        self.pushButton_analysis = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_analysis.setGeometry(QtCore.QRect(1180, 30, 100, 30))
        self.pushButton_analysis.setObjectName("pushButton_analysis")

        self.TextEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.TextEdit.setGeometry(QtCore.QRect(200, 200, 320, 150))
        self.TextEdit.setObjectName("TextEdit")

        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(200, 100, 110, 50))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.comboBox_data = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_data.setGeometry(QtCore.QRect(800, 30, 130, 30))
        self.comboBox_data.setObjectName("comboBox_data")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")
        self.comboBox_data.addItem("")

        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)

        self.label_sen1 = QtWidgets.QLabel(self.centralwidget)
        self.label_sen1.setGeometry(QtCore.QRect(70, 250, 101, 41))
        self.label_sen1.setFont(font)
        self.label_sen1.setObjectName("sentence1")

        self.label_sen2 = QtWidgets.QLabel(self.centralwidget)
        self.label_sen2.setGeometry(QtCore.QRect(70, 200, 101, 41))
        self.label_sen2.setFont(font)
        self.label_sen2.setObjectName("sentence2")

        self.label_sen3 = QtWidgets.QLabel(self.centralwidget)
        self.label_sen3.setGeometry(QtCore.QRect(95, 400, 101, 41))
        self.label_sen3.setFont(font)
        self.label_sen3.setObjectName("sentence3")

        self.label_sen4 = QtWidgets.QLabel(self.centralwidget)
        self.label_sen4.setGeometry(QtCore.QRect(70, 100, 101, 41))
        self.label_sen4.setFont(font)
        self.label_sen4.setObjectName("sentence4")

        self.label_sen5 = QtWidgets.QLabel(self.centralwidget)
        self.label_sen5.setGeometry(QtCore.QRect(680, 30, 101, 31))
        self.label_sen5.setFont(font)
        self.label_sen5.setObjectName("sentence5")

        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(650, 80, 790, 470))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setStyleSheet("selection-background-color:pink")
        self.tableWidget.raise_()

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_main.clicked.connect(self.btn_main_Press_clicked)
        self.pushButton_exit.clicked.connect(MainWindow.close)
        self.pushButton_open.clicked.connect(self.creat_table_show)
        self.pushButton_pic.clicked.connect(self.picture)
        # self.pushButton_export.clicked.connect()
        self.pushButton_analysis.clicked.connect(self.analysis)
        self.pushButton_introduction.clicked.connect(self.introduction)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle('感知差识别GUI')
        MainWindow.setWindowIcon(QtGui.QIcon('panda'))
        self.pushButton_main.setText(_translate("MainWindow", "感知差识别"))
        self.pushButton_exit.setText(_translate("MainWindow", "退出"))
        self.pushButton_pic.setText(_translate("MainWindow", "降维并聚类后的图片"))
        self.pushButton_introduction.setText(_translate("MainWindow", "使用说明"))
        self.pushButton_open.setText(_translate("MainWindow", "显示"))
        self.pushButton_export.setText(_translate("MainWindow", "导出"))
        self.pushButton_analysis.setText((_translate("MainWindow", "指标详情")))
        self.label_sen1.setText(_translate("MainWindow", "识别数据"))
        self.label_sen2.setText(_translate("MainWindow", "识别结果"))
        self.label_sen3.setText(_translate("MainWindow", "识别"))
        self.label_sen4.setText(_translate("MainWindow", "降维算法"))
        self.label_sen5.setText(_translate("MainWindow", "数据选择"))
        self.comboBox.setItemText(0, _translate("MainWindow", "自编码器"))
        self.comboBox.setItemText(1, _translate("MainWindow", "主成分分析"))
        self.comboBox_data.setItemText(0, _translate("MainWindow", "打开任意execl"))
        self.comboBox_data.setItemText(1, _translate("MainWindow", "4-感知差识别数据"))
        self.comboBox_data.setItemText(2, _translate("MainWindow", "3-验证数据"))
        self.comboBox_data.setItemText(3, _translate("MainWindow", "2-问题小区数据"))
        self.comboBox_data.setItemText(4, _translate("MainWindow", "1-感知不明数据"))

    def btn_main_Press_clicked(self):
        Alg = self.comboBox.currentText()
        if Alg == "自编码器":
            string, ECI, time, name = mainencoder()
            self.TextEdit.setPlainText("识别结果："+string+"\n"+"\n"+'识别小区为：'+name+"\n"+"小区ECI为："+ECI+"\n"+"时间为："+time)
        else:
            string, ECI, time, name = mainpca()
            self.TextEdit.setPlainText("识别结果："+string+"\n"+"\n"+'识别小区为：'+name +"\n"+"小区ECI为：" + ECI + "\n" + "时间为：" + time)

    def introduction(self):
        self.haoN = Ui_introduction()
        self.haoN.show()

    def picture(self):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
        # 图片路径设置与图片加载
        icon = QtGui.QIcon(path)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setColumnCount(1)
        self.tableWidget.setRowCount(1)
        self.tableWidget.setColumnWidth(0, 790)
        self.tableWidget.setRowHeight(0, 470)
        self.tableWidget.setHorizontalHeaderLabels(['图片展示'])
        self.tableWidget.setVerticalHeaderLabels([''])
        self.tableWidget.setIconSize(QtCore.QSize(800, 600))
        item.setIcon(QtGui.QIcon(icon))
        # 将条目加载到相应行列中
        self.tableWidget.setItem(0, 0, item)


    def creat_table_show(self):
        file = self.comboBox_data.currentText()
        if file == '打开任意execl':
            root = tk.Tk()
            root.withdraw()
            path_openfile_name = filedialog.askopenfilename()
        else:
            path_openfile_name = 'F:\\PYTHON\\毕设\\'+file+'.xlsx'

        if len(path_openfile_name) > 0:
            # input_table = pd.read_excel(path_openfile_name, names=[''])
            input_table = pd.read_excel(path_openfile_name,
                              names=['小区名', 'ECI', '时间', '页面响应成功率', '页面响应时延ms', '页面显示成功率',
                                     '页面显示时延ms', '页面下载速率kbps', '移动视频初始播放成功率', '移动视频停顿次数每分钟',
                                     '停顿时长占比', '初始缓存时延ms', '流媒体速率kbps', '即时通信响应成功率',
                                     '即时通信响应时延', '移动业务游戏响应成功率', '移动业务游戏响应时延（ms）'])
            input_table = input_table.drop(index=[0])
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            input_table_header = input_table.columns.values.tolist()

            self.tableWidget.setColumnCount(input_table_colunms)
            self.tableWidget.setRowCount(input_table_rows)
            self.tableWidget.setHorizontalHeaderLabels(input_table_header)
            self.tableWidget.setVerticalHeaderLabels([str(i) for i in range(168)])
            self.tableWidget.setColumnWidth(0, 130)
            self.tableWidget.setRowHeight(0, 40)

            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]

                    input_table_items = str(input_table_items_list)
                    newItem = QtWidgets.QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.tableWidget.setItem(i, j, newItem)

        else:
            self.centralWidget.show()

    def analysis(self):
        text = self.comboBox_data.currentText()
        if text == '打开任意execl':
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename()
        else:
            path = 'F:\\PYTHON\\毕设\\' + text + '.xlsx'
        df = pd.read_excel(path, names=['小区名', 'ECI', '时间', '页面响应成功率', '页面响应时延ms', '页面显示成功率',
                                               '页面显示时延ms', '页面下载速率kbps', '移动视频初始播放成功率', '移动视频停顿次数每分钟',
                                               '停顿时长占比', '初始缓存时延ms', '流媒体速率kbps', '即时通信响应成功率',
                                               '即时通信响应时延', '移动业务游戏响应成功率', '移动业务游戏响应时延（ms）'])
        df.drop(index=[0], inplace=True)
        df.drop(['小区名', 'ECI', '时间'], inplace=True, axis='columns')
        mean_ = df.mean(axis=0, skipna=True)
        max_ = df.max(axis=0, skipna=True)
        min_ = df.min(axis=0, skipna=True)
        median_ = df.median(axis=0, skipna=True)
        std_ = df.std(axis=0, skipna=True)

        mean = mean_.values.tolist()
        max = max_.values.tolist()
        min = min_.values.tolist()
        median = median_.values.tolist()
        std = std_.values.tolist()

        rows = len(mean)
        clos = 5

        self.tableWidget.setColumnCount(clos)
        self.tableWidget.setRowCount(rows)
        self.tableWidget.setColumnWidth(0, 130)
        self.tableWidget.setRowHeight(0, 40)
        title = ['平均值', '最大值', '最小值', '中位数', '方差']
        ROW = ['页面响应成功率', '页面响应时延ms', '页面显示成功率', '页面显示时延ms',
               '页面下载速率kbps', '移动视频初始播放成功率', '移动视频停顿次数每分钟',
               '停顿时长占比', '初始缓存时延ms', '流媒体速率kbps', '即时通信响应成功率',
               '即时通信响应时延', '移动业务游戏响应成功率', '移动业务游戏响应时延（ms）']
        self.tableWidget.setHorizontalHeaderLabels(title)
        self.tableWidget.setVerticalHeaderLabels(ROW)

        for j in range(rows):
            mean__ = round(mean[j], 2)
            max__ = max[j]
            min__ = min[j]
            median__ = median[j]
            std__ = round(std[j], 2)
            self.tableWidget.setItem(j, 0, QtWidgets.QTableWidgetItem(str(mean__)))
            self.tableWidget.setItem(j, 1, QtWidgets.QTableWidgetItem(str(max__)))
            self.tableWidget.setItem(j, 2, QtWidgets.QTableWidgetItem(str(min__)))
            self.tableWidget.setItem(j, 3, QtWidgets.QTableWidgetItem(str(median__)))
            self.tableWidget.setItem(j, 4, QtWidgets.QTableWidgetItem(str(std__)))


class Ui_introduction(QtWidgets.QWidget):
    def __init__(self):
        super(Ui_introduction, self).__init__()
        self.setObjectName("shuoming")
        self.setEnabled(True)
        self.resize(500, 200)
        self.setWindowTitle("课题的使用说明")
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(15, 0, 500, 200))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label.setText("使用说明：\n"
                           "1.首先在\"降维算法\"处选择降维算法，点击\"感知差识别\"\n"
                           "2.在\"识别数据\"处显示感知差识别数据\n"
                           "3.在\"识别结果\"处显示感知差识别结果\n"
                           "4.点击\"降维和聚类后的图片\"可以展示降维效果\n"
                           "5.点击\"数据选择\"和\"显示\"可以显示数据\n"
                           "6.点击\"指数详情\"可以查看数据的各项指标\n"
                           "7.点击\"退出\"退出GUI界面")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

