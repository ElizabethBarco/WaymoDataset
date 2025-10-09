import sys
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Get the filename from the first command-line argument
# sys.argv[0] is the script name itself, sys.argv[1] is the first argument
if len(sys.argv) < 2:
    print("❌ Error: Please provide a filename as an argument.")
    sys.exit(1)

FILENAME = sys.argv[1]

print(f"--- Processing file: {FILENAME} ---")

try:
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    # ... (rest of your parsing and analysis code) ...
    print("✅ Successfully processed file.")
except Exception as e:
    print(f"❌ Error processing file {FILENAME}: {e}")