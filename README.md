# napari-sperm-measure

A napari plugin for measuring sperm cell lengths in microscopy images. Built for the Fall 2024 CSE 554 course at Washington University in St. Louis.

## Team Members
Xinhang (Owen) Ma, 
Ziqiao (Aurora) Jiao

## Description

This plugin provides an interactive interface for measuring the length of sperm cells in microscopy images. It features:

- Interactive cell tracing with flood-fill algorithm
- Adaptive preprocessing with adjustable parameters
- Real-time measurement feedback
- Eraser tool for refinement
- Ground truth comparison
- Works for multiple microscopy image condition levels
- Save/export capabilities

## Features

- **Image Preprocessing**: Adjustable parameters for optimal cell detection
- **Interactive Tracing**: Click-based cell body tracing with flood fill
- **Measurement Tools**: Automatic length calculation in micrometers
- **Eraser Mode**: Clean up trace artifacts and noise
- **Multiple Difficulty Levels**: Easy, Medium, and Hard image sets
- **Ground Truth Comparison**: Compare measurements with known values
- **Keyboard Shortcuts**: Quick access to core functions

## Requirements

See `requirements.txt` for a complete list of dependencies. Main requirements include:

- Python >=3.11
- napari
- numpy
- opencv-python
- pyqt
  
## Installation

We highly recommend using a virtual environment to manage dependencies. We suggest using `conda`:
```terminal
conda create -n napari-sperm-measure python=3.11
```
Activate the environment:
```terminal
conda activate napari-sperm-measure
```
Clone this repository and navigate to the root directory:
```terminal
git clone https://github.com/xhOwenMa/napari-sperm-measure.git
cd napari-sperm-measure
```
Install the dependencies:
```terminal
pip install -r requirements.txt
```
You can then install this package via pip, run:
```terminal
pip install napari-sperm-measure
```

## Usage

We provide an example script to run our plugin with some sample images. To run the plugin, simply execute the following command in your terminal:
```terminal
python plugin.py
```
Alternatively, you can create our plugin in your own script (note that you will need to implement your own data loading logic):
```python
import napari
from napari_sperm_measure import create_sperm_measure_widget
from napari_sperm_measure.utils import create_measurement_window

viewer = napari.Viewer()
widget = create_sperm_measure_widget()
widget_window = create_measurement_window(widget)
widget_window.show()

napari.run()
```

Keyboard Shortcuts:
- `P`: Preprocess image
- `T`: Start tracing cell body
- `Esc`: Cancel tracing
- `M`: Measure cell length
- `S`: Save current layer image
- `H`: Show help

## Issues

If you encounter any problems, please file an issue along with a detailed description.

## License

This project is licensed under the terms of the MIT license.

## Acknowledgements

This project was developed as part of the Fall 2024 CSE 554 course at Washington University in St. Louis. We would like to thank our instructor, Professor [Tao Ju](https://www.cs.wustl.edu/~taoju/), for his support. We would also like to thank the [napari](https://napari.org/) team for their excellent software.