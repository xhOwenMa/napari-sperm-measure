from ._version import __version__

from .widget import SpermMeasureWidget, create_sperm_measure_widget

# Add this manifest dictionary
manifest = {
    'name': 'napari-sperm-measure',
    'display_name': 'Sperm Cell Measurement',
    'contributions': {
        'commands': [{
            'id': 'napari-sperm-measure.widget',
            'python_name': 'napari_sperm_measure.widget:create_sperm_measure_widget',
            'title': 'Measure Sperm Cell Length'
        }],
        'widgets': [{
            'command': 'napari-sperm-measure.widget',
            'display_name': 'Sperm Measure'
        }]
    }
}

@property
def napari_experimental_provide_dock_widget():
    return [create_sperm_measure_widget]