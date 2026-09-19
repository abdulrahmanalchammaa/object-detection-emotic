import csv
import shutil
import os

# Set the source and destination directories
src_dir = 'S:\object_detection\emotic\mscoco\images'
dst_dir = 'S:\object_detection\photos'

# Create the destination directory if it does not exist
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# Open the CSV file
with open('data.csv', 'r') as csv_file:
    # Read the file names from the CSV file
    file_names = csv.reader(csv_file)
    # Iterate through the file names
    for file_name in file_names:
        # Construct the full file path
        file_path = os.path.join(src_dir, file_name[0])
        # Copy the file to the destination directory
        shutil.copy(file_path, dst_dir)