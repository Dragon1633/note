# Pyside笔记

需要导入的库

```python
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton,  QPlainTextEdit,QMessageBox
'''
QtWidgets 专门负责Qt里控件窗口的图形功能
下属类：
    QApplication    应用程序
    QMainWindow     主窗口
    QPushButton     按钮
    QPlainTextEdit  纯文本的编辑框
    QMessageBox     弹出对话框
'''
```

程序启动

```python
app = QApplication([])  # （创建这个对象）提供整个图形界面程序的底层管理功能
starts=Stats()          # 启动界面程序(ui中类的名字)
starts.window.show()    # 执行窗口程序的展示
app.exec_()     		# 进入该程序的事件处理循环，等待用户的输入
```

一些操作

```python
# 按钮点击执行操作
self.button.clicked.connect(self.handleCalc)
# 读取文本框信息
info = self.textEdit.toPlainText()
for line in info.splitlines():
            if not line.strip():
                continue
            parts = line.split(' ')	            # 去掉列表中的空字符串内容
            parts = [p for p in parts if p]

```





pyQt6

```python
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6 import uic


if __name__ == '__main__':
    app = QApplication(sys.argv)			# sys.arvg为获取当前py文件的地址，实例化程序
    ui = uic.loadUi("./ui/test.ui")
    ui.show()
    sys.exit(app.exec())
```

```python
myLabel: QLabel = ui.label 		 # 获取label对象
print(myLabel.text())			 # 打印label文本值
```

**QTextEdit多行富文本框控件**

​		setPlainText(） 设置文本内容

​		toPlainText()	 获取文本内容

​		clear() 				清除所有内容

**QPlainTextEdit纯文本控件**

QPlainTextEdit纯文本控件，主要用来显示多行的文本内容

​		setPlainText(） 	设置文本内容

​		toPlainText() 		获取文本内容

​		clear() 		清除所有内容

```python
myPlainTextEdit: QPlainTextEdit = ui.plainTextEdit 				# plainTextEdit控件的对象名称
myPlainTextEdit.setPlainText("6666")
print(myPlainTextEdit.toPlainText())
```

**QPushButton按钮控件**

QPushButton是PyQt6中最常用的控件之一，它被称为按钮控件，允许用户通过单击来执行操作。QPushButton控件既可以显示文本，也可以显示图像，当该控件被单击时，它看起来像是被按下，然后被释放。



QTDesigner

![image-20240308114804894](D:\tool\typora\image\image-20240308114804894.png)

## 从一个窗口跳转到另外一个窗口

```python

class Window2(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('窗口2')
    
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('窗口1')    
        button = QtWidgets.QPushButton('打开新窗口')
        button.clicked.connect(self.open_new_window)

    def open_new_window(self):          # 实例化另外一个窗口
        self.window2 = Window2()        # 显示新窗口
        self.window2.show()		        # 关闭自己
        self.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

