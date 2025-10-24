import sys
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import numpy as np
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = './waymo_dataset/results'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
if len(sys.argv) < 2:
    print("Usage: python explore_proto.py <path_to_tfrecord>")
    sys.exit(1)

filename = sys.argv[1]
print(f"=== Exploring: {filename} ===\n")

try:
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    data_iter = dataset.as_numpy_iterator()
    print("✓ Successfully loaded dataset")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# ============================================================================
# EXPLORATION
# ============================================================================
frame_count = 0
exploration_data = {
    'filename': filename,
    'timestamp': datetime.now().isoformat(),
    'frames': []
}

for bytes_example in data_iter:
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(bytes_example)
    frame_count += 1
    
    print(f"\n{'='*80}")
    print(f"FRAME {frame_count}")
    print(f"{'='*80}")
    
    frame_info = {'frame_id': frame_count}
    
    # ========================================================================
    # FRAME CONTEXT
    # ========================================================================
    print(f"\n--- FRAME CONTEXT ---")
    if data.HasField('frame'):
        frame = data.frame
        print(f"Frame name: {frame.context.name}")
        frame_info['frame_name'] = frame.context.name
        print(f"Timestamp (microseconds): {frame.timestamp_micros}")
        frame_info['timestamp_micros'] = frame.timestamp_micros
        
        # Cameras
        print(f"\nCameras available: {len(frame.images)}")
        frame_info['num_cameras'] = len(frame.images)
        for i, img in enumerate(frame.images):
            print(f"  Camera {i}: name={img.name}, size={len(img.image)} bytes")
        
        # Calibrations
        print(f"\nCamera calibrations available: {len(frame.context.camera_calibrations)}")
        frame_info['num_calibrations'] = len(frame.context.camera_calibrations)
        for i, cal in enumerate(frame.context.camera_calibrations):
            print(f"  Calibration {i}: width={cal.width}, height={cal.height}")
    else:
        print("No frame data")
    
    # ========================================================================
    # FUTURE STATES (Trajectory prediction targets)
    # ========================================================================
    print(f"\n--- FUTURE STATES (t=0 to 5s, @4Hz) ---")
    if data.HasField('future_states'):
        future = data.future_states
        num_future = len(future.pos_x)
        print(f"Number of future waypoints: {num_future}")
        frame_info['future_states_count'] = num_future
        
        if num_future > 0:
            print(f"Future positions (x, y, z):")
            for i in range(min(5, num_future)):
                print(f"  Waypoint {i}: x={future.pos_x[i]:.2f}, y={future.pos_y[i]:.2f}, z={future.pos_z[i]:.2f}")
            if num_future > 5:
                print(f"  ... ({num_future - 5} more waypoints)")
        else:
            print("No future states available")
    else:
        print("No future_states field")
    
    # ========================================================================
    # PAST STATES (Historical motion data)
    # ========================================================================
    print(f"\n--- PAST STATES (t=-4s to 0, @4Hz) ---")
    if data.HasField('past_states'):
        past = data.past_states
        num_past = len(past.pos_x)
        print(f"Number of past waypoints: {num_past}")
        frame_info['past_states_count'] = num_past
        
        if num_past > 0:
            print(f"\nPast positions (x, y, z):")
            for i in range(min(5, num_past)):
                z_val = past.pos_z[i] if i < len(past.pos_z) else 'N/A'
                if isinstance(z_val, (int, float)):
                    print(f"  {i}: pos=({past.pos_x[i]:.2f}, {past.pos_y[i]:.2f}, {z_val:.2f})")
                else:
                    print(f"  {i}: pos=({past.pos_x[i]:.2f}, {past.pos_y[i]:.2f}, {z_val})")
            
            print(f"\nPast velocities (vx, vy) m/s:")
            for i in range(min(5, num_past)):
                if i < len(past.vel_x) and i < len(past.vel_y):
                    print(f"  {i}: vel=({past.vel_x[i]:.2f}, {past.vel_y[i]:.2f})")
            
            print(f"\nPast accelerations (ax, ay) m/s²:")
            for i in range(min(5, num_past)):
                if i < len(past.accel_x) and i < len(past.accel_y):
                    print(f"  {i}: accel=({past.accel_x[i]:.2f}, {past.accel_y[i]:.2f})")
            
            if num_past > 5:
                print(f"  ... ({num_past - 5} more states)")
            
            # Statistics
            print(f"\nPast motion statistics:")
            vel_x_array = np.array(past.vel_x) if len(past.vel_x) > 0 else np.array([0])
            vel_y_array = np.array(past.vel_y) if len(past.vel_y) > 0 else np.array([0])
            speed = np.sqrt(vel_x_array**2 + vel_y_array**2)
            print(f"  Speed: min={speed.min():.2f}, max={speed.max():.2f}, mean={speed.mean():.2f} m/s")
            
            accel_x_array = np.array(past.accel_x) if len(past.accel_x) > 0 else np.array([0])
            accel_y_array = np.array(past.accel_y) if len(past.accel_y) > 0 else np.array([0])
            print(f"  Accel X: min={accel_x_array.min():.2f}, max={accel_x_array.max():.2f} m/s²")
            print(f"  Accel Y: min={accel_y_array.min():.2f}, max={accel_y_array.max():.2f} m/s²")
            
            # Jerk calculation
            if len(accel_x_array) > 1:
                jerk_x = np.diff(accel_x_array)
                jerk_y = np.diff(accel_y_array)
                print(f"  Jerk X: min={jerk_x.min():.2f}, max={jerk_x.max():.2f} m/s³")
                print(f"  Jerk Y: min={jerk_y.min():.2f}, max={jerk_y.max():.2f} m/s³")
            
            frame_info['motion_stats'] = {
                'speed_min': float(speed.min()),
                'speed_max': float(speed.max()),
                'speed_mean': float(speed.mean()),
                'accel_x_min': float(min(past.accel_x)),
                'accel_x_max': float(max(past.accel_x)),
                'accel_y_min': float(min(past.accel_y)),
                'accel_y_max': float(max(past.accel_y))
            }
        else:
            print("No past states available")
    else:
        print("No past_states field")
    
    # ========================================================================
    # INTENT
    # ========================================================================
    print(f"\n--- DRIVING INTENT ---")
    if data.HasField('intent'):
        intent_map = {
            0: 'UNKNOWN',
            1: 'GO_STRAIGHT',
            2: 'GO_LEFT',
            3: 'GO_RIGHT'
        }
        intent_name = intent_map.get(data.intent, 'UNKNOWN')
        print(f"Intent: {data.intent} ({intent_name})")
        frame_info['intent'] = intent_name
    else:
        print("No intent field")
    
    # ========================================================================
    # PREFERENCE TRAJECTORIES (Human-rated trajectories)
    # ========================================================================
    print(f"\n--- PREFERENCE TRAJECTORIES (Human-rated) ---")
    num_pref = len(data.preference_trajectories)
    print(f"Number of preference trajectories: {num_pref}")
    frame_info['num_preference_trajectories'] = num_pref
    
    if num_pref > 0:
        for i, traj in enumerate(data.preference_trajectories):
            score = traj.preference_score if traj.HasField('preference_score') else -1
            waypoints = len(traj.pos_x)
            print(f"  Trajectory {i}: score={score}, waypoints={waypoints}")
            if waypoints > 0:
                print(f"    First waypoint: ({traj.pos_x[0]:.2f}, {traj.pos_y[0]:.2f})")
    
    # ========================================================================
    # ALL AVAILABLE FIELDS SUMMARY
    # ========================================================================
    print(f"\n--- ALL AVAILABLE FIELDS ---")
    fields = {
        'frame': data.HasField('frame'),
        'future_states': data.HasField('future_states'),
        'past_states': data.HasField('past_states'),
        'intent': data.HasField('intent'),
        'preference_trajectories': len(data.preference_trajectories) > 0
    }
    for field, available in fields.items():
        status = "✓ YES" if available else "✗ NO"
        print(f"  {field}: {status}")
    
    frame_info['available_fields'] = fields
    exploration_data['frames'].append(frame_info)
    
    # Stop after 10 frames for exploration
    if frame_count >= 10:
        print(f"\n{'='*80}")
        print("Stopping after 10 frames for exploration purposes")
        print(f"{'='*80}")
        break

# ============================================================================
# SAVE EXPLORATION RESULTS
# ============================================================================
output_file = f"{OUTPUT_DIR}/exploration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w') as f:
    json.dump(exploration_data, f, indent=2)

print(f"\n✓ Exploration data saved to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total frames explored: {frame_count}")
print(f"\nData types found:")
print(f"  ✓ Frame data (images + calibrations)")
print(f"  ✓ Past motion states (position, velocity, acceleration)")
print(f"  ✓ Future trajectory targets")
print(f"  ✓ Driving intent labels")
print(f"  ✓ Human-rated preference trajectories")
print(f"\nKey insights:")
print(f"  - Past states: Historical motion data for edge case detection")
print(f"  - Future states: Trajectory targets for prediction")
print(f"  - Intent: Driving action labels")
print(f"  - Preference trajectories: Human-rated alternative paths")
print(f"{'='*80}\n")
