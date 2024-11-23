# script to upload folder of images.

#!/bin/bash

# Folder containing the images
IMAGE_FOLDER="images/"

# API endpoint
API_URL="http://127.0.0.1:8000/upload_image/"

# Loop through all image files in the folder
for image in "$IMAGE_FOLDER"/*; do
  # Check if the file is indeed a file (to avoid errors with directories)
  if [[ -f "$image" ]]; then
    echo "Uploading $image..."
    curl -X POST "$API_URL" -F "file=@$image"
  fi
done