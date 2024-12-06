"""
Main widget module for the napari-sperm-measure plugin.
Provides the SpermMeasureWidget class that handles all UI interactions and measurement operations.
"""
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                          QComboBox, QHBoxLayout, QSpinBox, QSlider,
                          QFileDialog, QGridLayout, QMessageBox, QProgressBar, QShortcut, QToolButton)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence
import napari
import cv2
import numpy as np
import os
import json
import datetime
from .data_handling import DataManager
from .measurement import initial_preprocessing, trace_cell, measure_cell

class SpermMeasureWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.data_manager = DataManager()
        self.preprocessing_results = None
        self.point_selection_enabled = False
        self.mouse_callback = None
        self.accumulated_mask = None
        self.is_preprocessing = False
        self.is_erase_mode = False  # Add this new state variable
        self.erase_radius = 10  # Default erase radius
        
        # Set up the user interface
        self.setLayout(QVBoxLayout())
        
        # Add help button
        self.help_btn = QPushButton("?")
        self.help_btn.setFixedSize(25, 25)
        self.help_btn.clicked.connect(self._show_help)
        self.help_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border-radius: 12px;
                font-weight: bold;
            }
        """)
        self.layout().insertWidget(0, self.help_btn)
        
        # Add title label
        title_label = QLabel("Sperm Cell Measurement")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2196F3;")
        title_label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(title_label)
        
        # Add processing controls first
        self._setup_processing_controls()
        
        # Add image selection controls
        self._setup_image_controls()
        
        # Add file name label
        self.filename_label = QLabel()
        self.filename_label.setStyleSheet("""
            font-size: 11pt; 
            color: #666;
            padding: 5px;
            background-color: #f5f5f5;
            border-radius: 3px;
        """)
        self.filename_label.setWordWrap(True)
        self.layout().addWidget(self.filename_label)
        
        # Add ground truth label
        self.ground_truth_label = QLabel()
        self.ground_truth_label.setStyleSheet("font-size: 11pt; color: #4CAF50;")
        self.ground_truth_label.setWordWrap(True)
        self.layout().addWidget(self.ground_truth_label)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.layout().addWidget(self.progress_bar)
        
        # Add status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        self.layout().addWidget(self.status_label)
        
        # Set minimum size
        self.setMinimumWidth(300)
        
        # Add tooltips
        self._add_tooltips()
        
        # Add keyboard shortcuts
        self._setup_shortcuts()
        
        # Initialize with first image
        self._load_current_image()

    def _setup_image_controls(self):
        """Setup controls for image selection"""
        # Create control layout
        control_layout = QVBoxLayout()
        
        # Difficulty selection
        diff_layout = QHBoxLayout()
        diff_label = QLabel("Difficulty:")
        self.diff_combo = QComboBox()
        self.diff_combo.addItems(['Easy', 'Medium', 'Hard'])
        self.diff_combo.currentTextChanged.connect(self._on_difficulty_changed)
        diff_layout.addWidget(diff_label)
        diff_layout.addWidget(self.diff_combo)
        control_layout.addLayout(diff_layout)
        
        # Image navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self._on_previous_click)
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(0)
        self.image_spin.valueChanged.connect(self._on_image_index_changed)
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self._on_next_click)
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.image_spin)
        nav_layout.addWidget(self.next_btn)
        control_layout.addLayout(nav_layout)
        
        # Add to main layout
        self.layout().addLayout(control_layout)
        
        # Update image count for initial difficulty
        self._update_image_count()

    def _setup_processing_controls(self):
        """Setup controls for processing"""
        process_layout = QVBoxLayout()
        
        # Add parameter descriptions section
        param_desc_label = QLabel("""
        <b>Preprocessing Parameters:</b><br>
        <b>Block Size:</b> Size of pixel neighborhood for adaptive thresholding<br>
        <b>C Value:</b> Constant subtracted from mean, affects threshold sensitivity<br>
        <b>Kernel Size:</b> Size of area for morphological operations<br>
        <b>Iterations:</b> Number of times to apply morphological operations<br>
        <br>
        <b>Tips:</b><br>
        <b>Block Size:</b> Larger for better structure, but may create more gaps along the cell body<br>
        <b>C Value:</b> Increase this when the the picture has low contrast<br>
        <b>Kernel Size:</b> Larger values help close gaps, smaller values preserve detail<br>
        <b>Iterations:</b> More iterations smooth the result, fewer iterations preserve detail<br>
        """)
        param_desc_label.setStyleSheet("""
            font-size: 10pt;
            color: #666;
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
        """)
        param_desc_label.setWordWrap(True)
        process_layout.addWidget(param_desc_label)
        
        # Create a grid layout for sliders
        params_grid = QGridLayout()
        
        # Helper function to create info button
        def create_info_button(tooltip):
            info_btn = QToolButton()
            info_btn.setText("?")
            info_btn.setToolTip(tooltip)
            info_btn.setStyleSheet("""
                QToolButton {
                    border: none;
                    border-radius: 10px;
                    background-color: #2196F3;
                    color: white;
                    font-weight: bold;
                    padding: 2px;
                    width: 20px;
                    height: 20px;
                }
                QToolButton:hover {
                    background-color: #1976D2;
                }
                QToolTip {
                    background-color: #424242;
                    color: white;
                    border: 1px solid #757575;
                    padding: 5px;
                    font-size: 11pt;
                }
            """)
            return info_btn

        # Block Size Slider with info button
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("Block Size:"))
        block_layout.addWidget(create_info_button(
            "Larger values work better for images with varying brightness.\n"
            "Smaller values are better for uniform lighting."
        ))
        params_grid.addLayout(block_layout, 0, 0)
        
        # C Value Slider with info button
        c_value_layout = QHBoxLayout()
        c_value_layout.addWidget(QLabel("C Value:"))
        c_value_layout.addWidget(create_info_button(
            "Negative values make thresholding more aggressive.\n"
            "Positive values make it more lenient.\n"
            "Adjust based on image contrast."
        ))
        params_grid.addLayout(c_value_layout, 1, 0)
        
        # Kernel Size Slider with info button
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size:"))
        kernel_layout.addWidget(create_info_button(
            "Larger values help connect broken cell parts.\n"
            "Smaller values preserve fine details."
        ))
        params_grid.addLayout(kernel_layout, 2, 0)
        
        # Iterations Slider with info button
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Iterations:"))
        iter_layout.addWidget(create_info_button(
            "More iterations smooth the result but may lose detail.\n"
            "Fewer iterations preserve detail but may leave noise."
        ))
        params_grid.addLayout(iter_layout, 3, 0)

        # Block Size Slider (must be odd)
        self.block_size_slider = QSlider(Qt.Horizontal)
        self.block_size_slider.setMinimum(3)
        self.block_size_slider.setMaximum(99)
        self.block_size_slider.setValue(51)
        self.block_size_slider.setSingleStep(2)
        self.block_size_label = QLabel("Block Size: 51")
        params_grid.addWidget(self.block_size_slider, 0, 1)
        params_grid.addWidget(self.block_size_label, 0, 2)
        
        # C Value Slider
        self.c_value_slider = QSlider(Qt.Horizontal)
        self.c_value_slider.setMinimum(-20)
        self.c_value_slider.setMaximum(20)
        self.c_value_slider.setValue(-3)
        self.c_value_label = QLabel("C: -3")
        params_grid.addWidget(self.c_value_slider, 1, 1)
        params_grid.addWidget(self.c_value_label, 1, 2)
        
        # Kernel Size Slider (must be odd)
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setMinimum(3)
        self.kernel_slider.setMaximum(51)
        self.kernel_slider.setValue(3)
        self.kernel_slider.setSingleStep(2)
        self.kernel_label = QLabel("Kernel: 3")
        params_grid.addWidget(self.kernel_slider, 2, 1)
        params_grid.addWidget(self.kernel_label, 2, 2)
        
        # Iterations Slider
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setMinimum(1)
        self.iter_slider.setMaximum(20)
        self.iter_slider.setValue(1)
        self.iter_label = QLabel("Iterations: 1")
        params_grid.addWidget(self.iter_slider, 3, 1)
        params_grid.addWidget(self.iter_label, 3, 2)
        
        # Hide all parameter controls initially
        self._set_params_visibility(False)
        
        # Add the grid to the layout
        process_layout.addLayout(params_grid)
        
        # Connect slider signals
        self.block_size_slider.valueChanged.connect(
            lambda v: self._update_preprocessing_for_slider('block_size', v))
        self.c_value_slider.valueChanged.connect(
            lambda v: self._update_preprocessing_for_slider('c_value', v))
        self.kernel_slider.valueChanged.connect(
            lambda v: self._update_preprocessing_for_slider('kernel_size', v))
        self.iter_slider.valueChanged.connect(
            lambda v: self._update_preprocessing_for_slider('iterations', v))
        
        # Update slider labels
        self.block_size_slider.valueChanged.connect(
            lambda v: self.block_size_label.setText(f"Block Size: {v}"))
        self.c_value_slider.valueChanged.connect(
            lambda v: self.c_value_label.setText(f"C: {v}"))
        self.kernel_slider.valueChanged.connect(
            lambda v: self.kernel_label.setText(f"Kernel: {v}"))
        self.iter_slider.valueChanged.connect(
            lambda v: self.iter_label.setText(f"Iterations: {v}"))

        # Preprocess Image button
        self.preprocess_btn = QPushButton("Preprocess Image")
        self.preprocess_btn.clicked.connect(self._on_preprocess_click)
        self.preprocess_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
        """)
        process_layout.addWidget(self.preprocess_btn)
        
        # 'Trace Cell Body' button
        self.trace_btn = QPushButton("Trace Cell Body")
        self.trace_btn.clicked.connect(self._on_trace_click)
        self.trace_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        process_layout.addWidget(self.trace_btn)

        # Add erase mode toggle button after trace button
        self.erase_btn = QPushButton("Erase Mode: Off")
        self.erase_btn.clicked.connect(self._toggle_erase_mode)
        self.erase_btn.setEnabled(False)  # Disabled by default
        self.erase_btn.setStyleSheet("""
            QPushButton {
                background-color: #9E9E9E;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #757575;
            }
        """)
        process_layout.addWidget(self.erase_btn)

        # Add erase size control after erase button
        erase_size_layout = QHBoxLayout()
        erase_size_layout.addWidget(QLabel("Erase Size:"))
        self.erase_size_slider = QSlider(Qt.Horizontal)
        self.erase_size_slider.setMinimum(1)
        self.erase_size_slider.setMaximum(50)
        self.erase_size_slider.setValue(10)
        self.erase_size_label = QLabel("10")
        self.erase_size_slider.valueChanged.connect(
            lambda v: (self.erase_size_label.setText(str(v)), setattr(self, 'erase_radius', v))
        )
        erase_size_layout.addWidget(self.erase_size_slider)
        erase_size_layout.addWidget(self.erase_size_label)
        
        # Hide erase size controls initially
        self.erase_size_slider.setVisible(False)
        self.erase_size_label.setVisible(False)
        process_layout.addLayout(erase_size_layout)

        # Add info button for erase size
        erase_size_layout.insertWidget(1, create_info_button(
            "Adjust eraser size:\n"
            "Smaller values (1-10) for precise cleanup\n"
            "Larger values (10-50) for removing larger areas"
        ))

        # 'Measure Cell Length' button
        self.measure_btn = QPushButton("Measure Cell Length")
        self.measure_btn.clicked.connect(self._on_measure_click)
        self.measure_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        process_layout.addWidget(self.measure_btn)

        # Export Data button
        self.export_btn = QPushButton("Export Data")
        self.export_btn.clicked.connect(self._on_export_click)
        self.export_btn.setEnabled(False)  # Initially disabled until measurement is done
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #388E3C;
            }
        """)
        process_layout.addWidget(self.export_btn)
        
        # Result label
        self.result_label = QLabel()
        self.result_label.setStyleSheet("font-size: 11pt; color: #2196F3;")
        self.result_label.setWordWrap(True)
        process_layout.addWidget(self.result_label)
        
        self.layout().addLayout(process_layout)

    def _set_params_visibility(self, visible):
        """Show/hide parameter controls"""
        self.block_size_slider.setVisible(visible)
        self.c_value_slider.setVisible(visible)
        self.kernel_slider.setVisible(visible)
        self.iter_slider.setVisible(visible)
        self.block_size_label.setVisible(visible)
        self.c_value_label.setVisible(visible)
        self.kernel_label.setVisible(visible)
        self.iter_label.setVisible(visible)

    def _update_preprocessing_for_slider(self, param_name, value):
        """Update preprocessing for a specific parameter change"""
        if not self.is_preprocessing or not self.viewer.layers:
            return

        # Get current values from all sliders
        params = {
            'block_size': self.block_size_slider.value(),
            'c_value': self.c_value_slider.value(),
            'kernel_size': self.kernel_slider.value(),
            'iterations': self.iter_slider.value()
        }

        # Ensure block_size and kernel_size are odd
        if param_name in ['block_size', 'kernel_size']:
            if value % 2 == 0:
                value += 1
                if param_name == 'block_size':
                    self.block_size_slider.setValue(value)
                else:
                    self.kernel_slider.setValue(value)

        # Update the specific parameter
        params[param_name] = value

        # Get the original image
        image_layer = self.viewer.layers[0]
        image_data = image_layer.data

        # Update preprocessing with current parameters
        self.stages = initial_preprocessing(
            image_data,
            iterations=params['iterations'],
            kernel_size=params['kernel_size'],
            block_size=params['block_size'],
            c_value=params['c_value']
        )

        # Update the viewer layers
        for stage_name, stage_image in self.stages.items():
            if stage_name in self.viewer.layers:
                self.viewer.layers[stage_name].data = stage_image
            else:
                self.viewer.add_image(
                    stage_image,
                    name=stage_name,
                    colormap='gray',
                    blending='translucent',
                    opacity=1.0
                )

    def _on_preprocess_click(self):
        """Handle Preprocess Image button click"""
        if not self.viewer.layers:
            return

        if not self.is_preprocessing:
            self._update_status("Adjusting preprocessing parameters...")
            # Start preprocessing mode
            self.is_preprocessing = True
            self._set_params_visibility(True)
            self.preprocess_btn.setText("Apply Preprocessing")
        else:
            self._update_status("Preprocessing complete")
            # End preprocessing mode
            self.is_preprocessing = False
            self._set_params_visibility(False)
            self.preprocess_btn.setText("Preprocess Image")

    def _on_export_click(self):
        """Export skeleton image and measurement data"""
        if 'Cell Skeleton' not in self.viewer.layers or not self.result_label.text():
            self._update_status("Please measure the cell length first")
            return
            
        try:
            # Create base export directory if it doesn't exist
            export_dir = "measurement_exports"
            os.makedirs(export_dir, exist_ok=True)
            
            # Create skeleton images directory
            skeleton_dir = os.path.join(export_dir, "skeleton_images")
            os.makedirs(skeleton_dir, exist_ok=True)
            
            # Get current image information
            difficulty = self.diff_combo.currentText().lower()
            current_index = self.image_spin.value()
            current_image_id = self.data_manager.get_current_image_id(difficulty, current_index)
            
            if not current_image_id:
                self._update_status("Error: Could not determine current image ID")
                return
                
            measured_length = float(self.result_label.text().split(":")[1].strip().replace("mm", ""))
            
            # Save skeleton image
            skeleton_image = self.viewer.layers['Cell Skeleton'].data
            image_filename = f"{current_image_id}_skeleton.png"
            image_path = os.path.join(skeleton_dir, image_filename)
            cv2.imwrite(image_path, skeleton_image)
            
            # Prepare measurement data
            measurement_data = {
                'imageId': current_image_id,
                'difficulty': difficulty,
                'measured_length': measured_length,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Load or create measurements file
            measurements_file = os.path.join(export_dir, "measurements.json")
            if os.path.exists(measurements_file):
                with open(measurements_file, 'r') as f:
                    measurements = json.load(f)
                
                # Check if measurement for this image exists
                updated = False
                for i, entry in enumerate(measurements):
                    if entry['imageId'] == current_image_id:
                        measurements[i] = measurement_data
                        updated = True
                        break
                        
                if not updated:
                    measurements.append(measurement_data)
            else:
                measurements = [measurement_data]
                
            # Save measurements file
            with open(measurements_file, 'w') as f:
                json.dump(measurements, f, indent=4)
                
            self._update_status(f"Exported data for {current_image_id}")
            
        except Exception as e:
            self._update_status(f"Error exporting data: {str(e)}")

    def _load_current_image(self):
        """Load the currently selected image"""
        difficulty = self.diff_combo.currentText().lower()
        index = self.image_spin.value()
        
        # Clear existing layers
        self.viewer.layers.clear()
        
        # Load and display new image
        image_data, image_name, ground_truth = self.data_manager.load_image(difficulty, index)
        self.viewer.add_image(image_data, name=image_name)
        
        # Update filename display
        self.filename_label.setText(f"File: {image_name}")
        
        # Update ground truth display
        if ground_truth is not None:
            self.ground_truth_label.setText(f"Ground Truth Length: {ground_truth:.2f}mm")
        else:
            self.ground_truth_label.setText("No ground truth available")
        
        self._update_navigation_buttons()
        self.preprocessing_results = None  # Reset preprocessing results

        # Disable buttons
        self.preprocess_btn.setEnabled(True)

        # Clear stored stages and ROI layer
        self.stages = None

        # Reset tracing state when image changes
        self._disable_point_selection()
        self.accumulated_mask = None

    def _update_image_count(self):
        """Update the maximum value for image selection based on current difficulty"""
        difficulty = self.diff_combo.currentText().lower()
        count = self.data_manager.get_image_count(difficulty)
        self.image_spin.setMaximum(count - 1)
        self._update_navigation_buttons()
        self._disable_point_selection()
        self.accumulated_mask = None

    def _update_navigation_buttons(self):
        """Update the enabled state of navigation buttons"""
        current_idx = self.image_spin.value()
        max_idx = self.image_spin.maximum()
        
        self.prev_btn.setEnabled(current_idx > 0)
        self.next_btn.setEnabled(current_idx < max_idx)

    def _on_difficulty_changed(self, difficulty):
        """Handle difficulty selection change"""
        self.image_spin.setValue(0)
        self._update_image_count()
        self._load_current_image()
        # Reset tracing state when difficulty changes
        self._disable_point_selection()
        self.accumulated_mask = None

    def _on_image_index_changed(self, index):
        """Handle image index change"""
        self._load_current_image()
        # Reset tracing state when image index changes
        self._disable_point_selection()
        self.accumulated_mask = None

    def _on_previous_click(self):
        """Handle previous button click"""
        self.image_spin.setValue(self.image_spin.value() - 1)

    def _on_next_click(self):
        """Handle next button click"""
        self.image_spin.setValue(self.image_spin.value() + 1)

    def _on_trace_click(self):
        """Handle Trace Cell Body button click"""
        if not self.point_selection_enabled:
            self._update_status("Click points along the cell body...")
            self.point_selection_enabled = True
            self.trace_btn.setText("Stop Tracing")
            self.trace_btn.setStyleSheet("""
                QPushButton {
                    background-color: #F44336;
                    color: white;
                    padding: 5px;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #D32F2F;
                }
            """)
            # Initialize the accumulated mask
            if 'Preprocessed Image' in self.viewer.layers:
                self.accumulated_mask = np.zeros_like(self.viewer.layers['Preprocessed Image'].data, dtype=np.uint8)
            else:
                # Ensure that preprocessing has been done
                self.accumulated_mask = np.zeros_like(self.viewer.layers[0].data, dtype=np.uint8)
            # Enable mouse click callback
            self.mouse_callback = self.viewer.mouse_drag_callbacks.append(self._on_click)
            self.erase_btn.setEnabled(True)  # Enable erase button
        else:
            self.erase_btn.setEnabled(False)  # Disable erase button
            self.is_erase_mode = False  # Reset erase mode
            self.erase_btn.setText("Erase Mode: Off")
            self._update_status("Tracing complete")
            self._disable_point_selection()

    def _disable_point_selection(self):
        """Disable point selection mode"""
        self.point_selection_enabled = False
        self.trace_btn.setText("Trace Cell Body")
        self.trace_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        if self.mouse_callback is not None:
            self.viewer.mouse_drag_callbacks.remove(self.mouse_callback)
            self.mouse_callback = None

    def _toggle_erase_mode(self):
        """Toggle between trace and erase modes"""
        if self.point_selection_enabled:
            self.is_erase_mode = not self.is_erase_mode
            # Show/hide erase size controls
            self.erase_size_slider.setVisible(self.is_erase_mode)
            self.erase_size_label.setVisible(self.is_erase_mode)
            
            if self.is_erase_mode:
                self.erase_btn.setText("Erase Mode: On")
                self.erase_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #F44336;
                        color: white;
                        padding: 5px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
                self._update_status(f"Click on areas to erase them (size: {self.erase_radius}px)")
            else:
                self.erase_btn.setText("Erase Mode: Off")
                self.erase_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #9E9E9E;
                        color: white;
                        padding: 5px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #757575;
                    }
                """)
                self._update_status("Click points along the cell body...")

    def _on_click(self, viewer, event):
        """Handle mouse click events"""
        if not self.point_selection_enabled or 'Preprocessed Image' not in self.viewer.layers:
            return
        
        # Get click coordinates
        data_coordinates = viewer.layers['Preprocessed Image'].world_to_data(event.position)
        y, x = int(data_coordinates[-2]), int(data_coordinates[-1])
        
        if self.is_erase_mode:
            # Erase mode: create a circular mask around the clicked point
            if self.accumulated_mask is not None:
                cv2.circle(self.accumulated_mask, (x, y), self.erase_radius, 0, -1)
                
                # Update the 'Cell Mask' layer
                if 'Cell Mask' in self.viewer.layers:
                    self.viewer.layers['Cell Mask'].data = self.accumulated_mask
                    self.viewer.layers['Cell Mask'].refresh()
        else:
            # Normal tracing mode
            # Get the preprocessed image
            image = self.viewer.layers['Preprocessed Image'].data
            
            # Perform flood fill
            cell_body = trace_cell(image, (y, x))
            
            # Accumulate the mask
            if self.accumulated_mask is None:
                self.accumulated_mask = np.zeros_like(image, dtype=np.uint8)
            self.accumulated_mask = np.maximum(self.accumulated_mask, cell_body)

            # Update or add the 'Cell Mask' layer
            if 'Cell Mask' in self.viewer.layers:
                self.viewer.layers['Cell Mask'].data = self.accumulated_mask
                self.viewer.layers['Cell Mask'].refresh()
            else:
                self.viewer.add_image(
                    self.accumulated_mask,
                    name='Cell Mask',
                    colormap='green',
                    blending='translucent',
                    opacity=0.5
                )

    def _on_measure_click(self):
        """measure the cell length using the accumulated mask"""
        if self.accumulated_mask is None:
            self._update_status("Please trace the cell body first")
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        def update_progress():
            current = self.progress_bar.value()
            if current < 100:
                self.progress_bar.setValue(current + 20)
            else:
                timer.stop()
                self.progress_bar.hide()
                # Perform actual measurement
                skeleton, length = measure_cell(self.accumulated_mask)
                if 'Cell Skeleton' in self.viewer.layers:
                    self.viewer.layers['Cell Skeleton'].data = skeleton
                    self.viewer.layers['Cell Skeleton'].refresh()
                else:
                    self.viewer.add_image(
                        skeleton,
                        name='Cell Skeleton',
                        colormap='red',
                        blending='additive',
                        opacity=1.0
                    )
                
                # Update the result label
                self.result_label.setText(f"Measured Cell Length: {length:.2f}mm")
                self._update_status(f"Measurement complete: {length:.2f}mm")
                
                # Enable export button
                self.export_btn.setEnabled(True)

        timer = QTimer(self)
        timer.timeout.connect(update_progress)
        timer.start(100)

    def _add_tooltips(self):
        """Add helpful tooltips to all controls"""
        self.preprocess_btn.setToolTip("Enhance the image for better cell detection (Shortcut: P)")
        self.trace_btn.setToolTip("Click points along the cell to trace its body (Shortcut: T)")
        self.measure_btn.setToolTip("Calculate the length of the traced cell (Shortcut: M)")
        self.export_btn.setToolTip("Export skeleton image and measurement data (Shortcut: S)")
        self.block_size_slider.setToolTip("Adjust the size of the local area for thresholding")
        self.c_value_slider.setToolTip("Fine-tune the threshold sensitivity")
        self.kernel_slider.setToolTip("Adjust the size of morphological operations")
        self.iter_slider.setToolTip("Set the number of iterations for processing")
        
        # Add detailed parameter tooltips
        self.block_size_slider.setToolTip(
            "Larger values work better for images with varying brightness.\n"
            "Smaller values are better for uniform lighting."
        )
        self.c_value_slider.setToolTip(
            "Negative values make thresholding more aggressive.\n"
            "Positive values make it more lenient.\n"
            "Adjust based on image contrast."
        )
        self.kernel_slider.setToolTip(
            "Larger values help connect broken cell parts.\n"
            "Smaller values preserve fine details."
        )
        self.iter_slider.setToolTip(
            "More iterations smooth the result but may lose detail.\n"
            "Fewer iterations preserve detail but may leave noise."
        )
        self.erase_btn.setToolTip(
            "Toggle between tracing and erasing modes.\n"
            "In erase mode, click on incorrectly traced areas to remove them."
        )
        self.erase_size_slider.setToolTip(
            "Adjust eraser size:\n"
            "Smaller values (1-10) for precise cleanup\n"
            "Larger values (10-50) for removing larger areas"
        )

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        QShortcut(QKeySequence('P'), self, self._on_preprocess_click)
        QShortcut(QKeySequence('T'), self, self._on_trace_click)
        QShortcut(QKeySequence('M'), self, self._on_measure_click)
        QShortcut(QKeySequence('S'), self, self._on_export_click)
        QShortcut(QKeySequence('H'), self, self._show_help)
        QShortcut(QKeySequence('Esc'), self, self._disable_point_selection)

    def _show_help(self):
        """Display help message"""
        help_text = """
        <h3>How to Use the Sperm Cell Measurement Tool</h3>
        <ol>
            <li>Select an image using the difficulty and navigation controls</li>
            <li>Click 'Preprocess Image' and adjust the parameters if needed:</li>
            <li>Click 'Trace Cell Body' and click points along the cell</li>
            <li>Click 'Measure Cell Length' to calculate the length</li>
        </ol>
        <p><b>Keyboard Shortcuts:</b></p>
        <ul>
            <li>P: Preprocess Image</li>
            <li>T: Trace Cell Body</li>
            <li>M: Measure Cell Length</li>
            <li>S: Save Current Layer</li>
            <li>H: Show Help</li>
            <li>Esc: Cancel Tracing</li>
        </ul>
        <p><b>Tracing Tips:</b></p>
        <ul>
            <li>Click along the cell body to trace it</li>
            <li>Use Erase Mode to remove incorrectly traced areas or noticeable noises connected to the cell body that are not removed during preprocessing</li>
            <li>Toggle on and off erase mode as needed</li>
        </ul>
        <p><b>Erasing Tips:</b></p>
        <ul>
            <li>Use small eraser size (1-10) for cleaning up noise connected to the cell</li>
            <li>Use larger eraser size (10-50) for removing misclicked areas</li>
            <li>Adjust eraser size as needed while erasing</li>
        </ul>
        """
        QMessageBox.information(self, "Help", help_text)

    def _update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)

def create_sperm_measure_widget():
    """Create the widget for measuring sperm cell length."""
    return SpermMeasureWidget(napari.current_viewer())