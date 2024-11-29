from typing import Optional
from qtpy.QtWidgets import QMainWindow


def create_measurement_window(widget: Optional[QMainWindow] = None) -> QMainWindow:
    """Creates and configures the measurement window"""
    window = QMainWindow()
    window.setWindowTitle('Sperm Cell Measurement')
    if widget:
        window.setCentralWidget(widget)
    window.resize(500, 800)
    return window