import sys
from PyQt5 import QtWidgets, QtCore, QtGui
# This will conflict with OpenCV2

class CalibrationOverlay(QtWidgets.QWidget):
    def __init__(self, points, dot_radius=10, display_duration=8):
        super().__init__()
        self.points = points
        self.dot_radius = dot_radius
        self.display_duration = display_duration  # seconds
        self.setup_overlay()

    def setup_overlay(self):
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_ShowWithoutActivating)

        # Get primary screen geometry in a cross-platform way
        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)
        self.showFullScreen()

        # Optional auto-close after duration (for demo)
        QtCore.QTimer.singleShot(
            self.display_duration * 1000, QtWidgets.QApplication.quit
        )

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(
            QtGui.QBrush(QtGui.QColor(0, 255, 0, 255))
        )  # Bright green dots
        painter.setPen(QtGui.QPen(QtCore.Qt.green, 2))

        for x, y in self.points:
            painter.drawEllipse(QtCore.QPointF(x, y), self.dot_radius, self.dot_radius)


def generate_grid(screen_width, screen_height, rows=3, cols=3):
    xs = [int(screen_width * i / (cols - 1)) for i in range(cols)]
    ys = [int(screen_height * j / (rows - 1)) for j in range(rows)]
    return [(x, y) for y in ys for x in xs]


def main():
    app = QtWidgets.QApplication(sys.argv)

    screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()

    points = generate_grid(screen_width, screen_height)

    overlay = CalibrationOverlay(points)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
