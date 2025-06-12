#!/usr/bin/env python3
import os
import json

# Path to the LR files
lr_path = "/nr/bamjo/projects/SuperAI/usr/sarmad/BlindSRSNF/load/opensrtest/100/lr_12"
output_file = "/nr/bamjo/projects/SuperAI/usr/sarmad/difffusr/lr_files_list.txt"

def save_lr_file_list():
    """
    Save a list of all LR files in the specified directory to a text file
    """
    if not os.path.exists(lr_path):
        print(f"Error: Path {lr_path} does not exist")
        return
    
    try:
        # Get list of files in the directory
        lr_files = [f for f in os.listdir(lr_path) if os.path.isfile(os.path.join(lr_path, f))]
        
        # Save list to a text file
        with open(output_file, 'w') as f:
            for file in lr_files:
                f.write(f"{file}\n")
        
        print(f"Successfully saved {len(lr_files)} file names to {output_file}")
    except Exception as e:
        print(f"Error saving file list: {e}")

if __name__ == "__main__":
    save_lr_file_list()
