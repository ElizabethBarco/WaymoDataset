import tensorflow as tf
# This is the new, correct import for the End-to-End dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

# Your sharded filename with a wildcard to read all pieces
FILENAME = '/waymo_dataset/training/test_202504211836-202504220845.tfrecord-*'

print(f"--- Attempting to load Waymo E2E dataset from: {FILENAME} ---")

try:
    # tf.io.matching_files finds all files that match the wildcard pattern
    filenames = tf.io.matching_files(FILENAME)
    dataset = tf.data.TFRecordDataset(filenames, compression_type='')

    frame_count = 0
    # Iterate over the first 5 records as a test
    for i, data in enumerate(dataset.take(5)):
        # This is the new, correct data object for parsing
        e2e_frame = wod_e2ed_pb2.E2EDFrame()
        e2e_frame.ParseFromString(data.numpy())
        frame_count += 1
        print(f"Successfully parsed frame {i+1} with context name: {e2e_frame.frame.context.name}")

    print(f"\n✅ Success! Loaded and parsed {frame_count} frames from the dataset.")

except Exception as e:
    print(f"❌ Error loading dataset: {e}")