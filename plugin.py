"""
Plugin entry point module.
Handles initialization of the viewer and measurement interface.
"""

from typing import Optional, Tuple
import napari
from napari_sperm_measure import create_sperm_measure_widget
from napari_sperm_measure.data_handling import DataManager
from napari_sperm_measure.utils import create_measurement_window
from qtpy.QtWidgets import QMainWindow

def initialize_viewer() -> Tuple[napari.Viewer, DataManager]:
    """Initializes the napari viewer and data manager"""
    viewer = napari.Viewer()
    data_manager = DataManager()
    return viewer, data_manager

def main():
    """Main entry point for the sperm measurement plugin"""
    viewer, data_manager = initialize_viewer()
    
    # Load sample image
    image_data, image_name, ground_truth = data_manager.load_image('easy', 0)
    
    # Configure layer name
    layer_name = f"Image: {image_name}"
    if ground_truth is not None:
        layer_name += f" (Ground Truth: {ground_truth:.2f}mm)"
    viewer.add_image(image_data, name=layer_name)
    
    # Setup measurement interface
    widget = create_sperm_measure_widget()
    widget_window = create_measurement_window(widget)
    widget_window.show()
    
    napari.run()

if __name__ == '__main__':
    main()