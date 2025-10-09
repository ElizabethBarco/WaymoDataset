#!/bin/bash

# --- Configuration ---
# The public GCS bucket for the Waymo E2E dataset
GCS_BUCKET="gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
# The local directory where we will temporarily store the data
LOCAL_DATA_DIR="./waymo_dataset/training"
# The directory inside the container
CONTAINER_DATA_DIR="/waymo_dataset/training"

# --- Main Loop ---

# 1. Get a list of all .tfrecord files in the GCS bucket
echo "Fetching list of files from GCS..."
# The 'gsutil ls' command lists the contents of the bucket
# We use 'head -n 5' to only process the first 5 files for this example.
# Remove ' | head -n 5' to process the entire dataset.
file_list=$(gsutil ls "$GCS_BUCKET"/*.tfrecord* | head -n 5)

# 2. Loop through each file in the list
for gcs_file_path in $file_list; do
    # Get just the filename from the full GCS path
    filename=$(basename "$gcs_file_path")
    local_file_path="$LOCAL_DATA_DIR/$filename"
    container_file_path="$CONTAINER_DATA_DIR/$filename"

    echo "----------------------------------------"
    echo "Processing: $filename"
    echo "----------------------------------------"

    # 3. Download the single file
    echo "Downloading..."
    gsutil cp "$gcs_file_path" "$local_file_path"

    # 4. Process the file using Docker Compose
    echo "Running analysis..."
    # 'docker-compose run' starts a one-off container, runs the command, and then stops.
    # The '--rm' flag automatically cleans up the container afterward.
    docker-compose run --rm waymo-e2e-loader python load_dataset.py "$container_file_path"

    # 5. Delete the file to save space
    echo "Cleaning up..."
    rm "$local_file_path"

    echo "Done with $filename."
done

echo "----------------------------------------"
echo "All files processed."
echo "----------------------------------------"