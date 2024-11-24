# Script to upload a hierarchy of images with labels as folder names.

#!/bin/bash

# Top-level folder containing the subfolders with images
IMAGE_FOLDER="images/"

# API endpoint
API_URL="http://127.0.0.1:8000/upload_image_dev/"

# Dataset ID
DSID=1  # Change this value as needed for different datasets

# Loop through all subfolders in the top folder
for label_folder in "$IMAGE_FOLDER"/*; do
  # Ensure it's a directory
  if [[ -d "$label_folder" ]]; then
    # Extract the folder name to use as the label
    LABEL=$(basename "$label_folder")
    
    # Loop through all image files in the subfolder
    for image in "$label_folder"/*; do
      # Check if the file is indeed a file (to avoid errors with directories)
      if [[ -f "$image" ]]; then
        echo "Uploading $image with DSID $DSID and label $LABEL..."
        curl -X POST "$API_URL?dsid=$DSID&label=$LABEL" -F "file=@$image"
      fi
    done
  fi
done