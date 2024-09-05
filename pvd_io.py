import os
import sys
from io import StringIO

# Capture and display cell output
class OutputCapture:
    def __init__(self):
        self.outputs = []
        self.original_stdout = sys.stdout
        self.buffer = StringIO()

    def write(self, data):
        self.original_stdout.write(data)
        self.buffer.write(data)

    def flush(self):
        self.original_stdout.flush()
        self.buffer.flush()

    def get_output(self):
        return self.buffer.getvalue()
    
# Function to get files and paths
def scan_directories(data_directory, min_size_bytes, filetype='.tif'):
    dataset_list = []
    session_list = []
    file_list = []

    for root, dirs, files in os.walk(data_directory):
        # Check if the current path contains "exclude" or "movement"
        if "exclude" in root.lower() or "movement" in root.lower() or "missing" in root.lower():
            continue  # Skip this directory

        for file in files:
            if file.endswith(filetype):
                file_path = os.path.join(root, file)

                # Check file size
                if os.path.getsize(file_path) < min_size_bytes:
                    continue  # Skip this file if it's too small

                path_parts = os.path.normpath(root).split(os.sep)
                if len(path_parts) >= 3:
                    dataset = path_parts[-2]
                    session = path_parts[-1]
                    dataset_list.append(dataset)
                    session_list.append(session)
                    file_list.append(file)

    return dataset_list, session_list, file_list