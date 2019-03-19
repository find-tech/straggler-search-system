
# !pip install PyQt5
import sys
import pickle
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot

class ResultViewer(object):
    """Result Viewer:

    Args:
        result list
    """
    def __init__(self, results):
        self.results = results
        
    def print(self):
        print(self.results)

    def load_result(self, path='../result.pkl'):
        with open(path, 'rb') as f:
            self.results = pickle.load(f)

    def save_result(self, path='../result.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
        
    def save_txt(self, path='../result.txt'):
        with open(path, 'w') as f:
            f.writelines(str(self.results))

    def show_web(self):
        print('web:')
        print(self.results)

    def show_gui(self):
        app = QApplication(sys.argv)
        ex = App(self.results)
        sys.exit(app.exec_())

class App(QWidget):

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.title = "PyQt5 table - pythonspot.com"
        self.left = 300
        self.top = 100
        self.width = 1200
        self.height = 800
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.createTable()

        # Add box layout, add table to box layout and add box layout to widget
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget) 
        self.setLayout(self.layout) 

        # Show widget
        self.show()

    def createTable(self):
        # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(len(self.data))
        self.tableWidget.setColumnCount(7)

        i = 0
        for line in self.data:
            # ここの表示を作りこむ
            self.tableWidget.setItem(i, 0, QTableWidgetItem(line['camera_id']))
            j = 1
            for people in line['found_people']:
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(people['score'])))
                j += 1
            i += 1

        self.tableWidget.move(0,0)

        # table selection change
        self.tableWidget.doubleClicked.connect(self.on_click)

    @pyqtSlot()
    def on_click(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

if __name__ == "__main__":
    rv = ResultViewer(None)
    rv.load_result()
    #rv.print()
    #rv.save_txt()
    rv.show_gui()