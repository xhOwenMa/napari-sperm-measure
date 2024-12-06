import os
from pathlib import Path
import pandas as pd
from typing import Dict, List
import numpy as np
from PIL import Image
import sys

class DataManager:
    def __init__(self, data_dir: Path = None):
        # Try different possible locations for the test data
        possible_paths = [
            data_dir if data_dir is not None else None,  # User provided path
            Path(__file__).parent.parent.parent / 'test_data' / 'sperm',  # Development path
            Path.cwd() / 'test_data' / 'sperm',  # Current working directory
        ]

        # Find first valid path
        self.test_data_dir = None
        for path in possible_paths:
            if path is not None and path.exists():
                self.test_data_dir = path
                break

        if self.test_data_dir is None:
            raise FileNotFoundError(
                "Could not find test_data directory. Please provide the correct path or ensure "
                "the test_data directory exists in one of the following locations:\n" +
                "\n".join(str(p) for p in possible_paths if p is not None)
            )

        # Fix the path handling for the Excel file
        self.ground_truth_file = str(self.test_data_dir / 'manual.xlsx')

        # Load ground truth data
        self.ground_truth_by_difficulty = self._load_ground_truth()
        
        # Initialize image directories
        self.easy_dir = self.test_data_dir / 'easy'
        self.medium_dir = self.test_data_dir / 'medium'
        self.hard_dir = self.test_data_dir / 'hard'

    def _load_ground_truth(self) -> Dict[str, Dict[str, float]]:
        """
        Load ground truth data from Excel file
        """
        try:
            # Try to import openpyxl explicitly to give better error message
            try:
                import openpyxl
            except ImportError:
                print("Error: openpyxl is not installed in the current Python environment")
                print(f"Current Python interpreter: {sys.executable}")
                print("\nTo fix this, try the following steps:")
                print("1. Open a terminal")
                print(f"2. Run: {sys.executable} -m pip install openpyxl")
                print("\nIf that doesn't work, you might be using a virtual environment.")
                print("Make sure to activate the correct environment before installing.")
                return {'hard': {}, 'medium': {}, 'easy': {}}

            # Load Excel file
            df = pd.read_excel(self.ground_truth_file, engine='openpyxl')
            
            # Initialize result dictionary
            result = {
                'hard': {},
                'medium': {},
                'easy': {}
            }
            
            # Track current difficulty
            current_difficulty = None            
            # Iterate through rows
            for index, row in df.iterrows():
                # Check if this row defines a difficulty
                if pd.notna(row.iloc[0]):  # If first column is not empty
                    current_difficulty = row.iloc[0].lower()
                    continue
                if current_difficulty and pd.notna(row['ImageID']):
                    image_id = str(row['ImageID']).strip()
                    length = float(row['Length.Manual.mm'])
                    result[current_difficulty][image_id] = length
            
            return result
            
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            import traceback
            traceback.print_exc()  # This will print the full error trace
            return {'hard': {}, 'medium': {}, 'easy': {}}

    def load_image(self, difficulty: str, image_index: int = 0) -> tuple:
        """
        Load an image from the specified difficulty folder
        """
        difficulty = difficulty.lower()
        
        # Select the appropriate directory
        if difficulty == 'easy':
            dir_path = self.easy_dir
        elif difficulty == 'medium':
            dir_path = self.medium_dir
        elif difficulty == 'hard':
            dir_path = self.hard_dir
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

        # Get list of jpg files
        image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
        
        if not image_files:
            raise FileNotFoundError(f"No jpg files found in {dir_path}")
        
        if image_index >= len(image_files):
            raise IndexError(f"Image index {image_index} out of range. Only {len(image_files)} images available.")
        
        # Load the image
        image_name = image_files[image_index]
        image_path = dir_path / image_name
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Get ground truth length if available
        image_id = image_name.split('.jpg')[0]  # Remove .jpg extension
        
        # Try to find matching ground truth
        ground_truth_length = None
        
        # First try exact match
        if image_id in self.ground_truth_by_difficulty[difficulty]:
            ground_truth_length = self.ground_truth_by_difficulty[difficulty][image_id]
        else:
            # Try matching without the "_20x" suffix for WT.C images
            if image_id.startswith('WT.C'):
                base_id = image_id.replace('_20x', '')
                if base_id in self.ground_truth_by_difficulty[difficulty]:
                    ground_truth_length = self.ground_truth_by_difficulty[difficulty][base_id]
        
        return image_array, image_name, ground_truth_length
    
    def get_current_image_id(self, difficulty: str, image_index: int) -> str:
        """Get the image ID for the current image without extension"""
        difficulty = difficulty.lower()
        
        # Select the appropriate directory
        if difficulty == 'easy':
            dir_path = self.easy_dir
        elif difficulty == 'medium':
            dir_path = self.medium_dir
        elif difficulty == 'hard':
            dir_path = self.hard_dir
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")

        # Get list of jpg files
        image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
        
        if not image_files or image_index >= len(image_files):
            return None
            
        # Get image name without extension
        image_name = image_files[image_index]
        image_id = image_name.split('.jpg')[0]
    
        return image_id
    
    def get_image_count(self, difficulty: str) -> int:
        """Get the number of images in a difficulty folder"""
        difficulty = difficulty.lower()
        if difficulty == 'easy':
            dir_path = self.easy_dir
        elif difficulty == 'medium':
            dir_path = self.medium_dir
        elif difficulty == 'hard':
            dir_path = self.hard_dir
        else:
            raise ValueError("Difficulty must be 'easy', 'medium', or 'hard'")
            
        return len([f for f in os.listdir(dir_path) if f.endswith('.jpg')])