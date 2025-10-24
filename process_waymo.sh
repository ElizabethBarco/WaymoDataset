#!/bin/bash

set -e  # Exit on error

# --- Configuration ---
GCS_BUCKET="gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
DOWNLOAD_DIR="./waymo_dataset/downloads"
RESULTS_DIR="./waymo_dataset/results"
CONTAINER_DOWNLOAD_DIR="/waymo_dataset/downloads"
CONTAINER_RESULTS_DIR="/waymo_dataset/results"

# --- Create directories ---
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$RESULTS_DIR"

# --- Build once ---
echo "Building Docker image..."
docker-compose build waymo-e2e-loader
echo "✓ Build complete"

# --- Main Loop ---
echo "Fetching list of files from GCS..."
file_list=$(gsutil ls "$GCS_BUCKET"/*.tfrecord* | head -n 5)

for gcs_file_path in $file_list; do
    filename=$(basename "$gcs_file_path")
    local_file_path="$DOWNLOAD_DIR/$filename"
    container_file_path="$CONTAINER_DOWNLOAD_DIR/$filename"

    echo "----------------------------------------"
    echo "Processing: $filename"
    echo "----------------------------------------"

    echo "Downloading to $DOWNLOAD_DIR..."
    gsutil -m cp "$gcs_file_path" "$local_file_path"

    echo "Running analysis..."
    docker-compose run --rm \
        -e RESULTS_DIR="$CONTAINER_RESULTS_DIR" \
        waymo-e2e-loader python load_dataset.py "$container_file_path"

    echo "Cleaning up downloaded file..."
    rm "$local_file_path"

    echo "Done with $filename."
done

echo "----------------------------------------"
echo "✓ All files processed."
echo "✓ Results saved to: $RESULTS_DIR"
echo "----------------------------------------"