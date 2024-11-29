import os
from pathlib import Path
import pandas as pd
from typing import Dict, List
import numpy as np
from PIL import Image

class DataManager:
    def __init__(self):
        # Get the root directory of the package
        self.root_dir = Path(__file__).parent.parent.parent
        self.test_data_dir = self.root_dir / 'test_data' / 'sperm'
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
            # Load Excel file directly without adding extension
            df = pd.read_excel(self.ground_truth_file)
            
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