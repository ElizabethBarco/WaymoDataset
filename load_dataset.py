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
    frame_count = 0
    # Iterate over the first 5 records as a test
    for i, data in enumerate(dataset.take(5)):
        # This is the new, correct data object for parsing
        e2e_frame = wod_e2ed_pb2.E2EDFrame()
        e2e_frame.ParseFromString(data.numpy())
        frame_count += 1
        print(f"Successfully parsed frame {i+1} with context name: {e2e_frame.frame.context.name}")
except Exception as e:
    print(f"❌ Error processing file {FILENAME}: {e}")