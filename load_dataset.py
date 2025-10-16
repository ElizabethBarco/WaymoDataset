import sys
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import numpy as np
import sqlite3
import os
import json
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use environment variable if set, otherwise use default
RESULTS_DIR = os.getenv('RESULTS_DIR', './waymo_dataset/results')

DB_PATH = os.path.join(RESULTS_DIR, 'edge_cases.db')
THRESHOLD_FILE = os.path.join(RESULTS_DIR, 'thresholds.json')

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Default industry-standard thresholds
DEFAULT_THRESHOLDS = {
    'hard_brake': -0.8,      # Moderate to hard braking
    'lateral': 0.6,          # Evasive maneuver (lateral G-force)
    'jerk': 0.4              # Sudden acceleration change
}

# ============================================================================
# DATABASE SETUP
# ============================================================================
def init_database(db_path):
    """Initialize SQLite database for edge case storage."""
    try:
        # Use check_same_thread=False for container compatibility
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for better concurrency
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edge_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                file_name TEXT,
                timestamp BIGINT,
                edge_case_type TEXT,
                severity REAL,
                speed_min REAL,
                speed_max REAL,
                accel_x_min REAL,
                accel_y_max REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        return conn
    except Exception as e:
        print(f"✗ Database initialization error: {e}")
        raise

def store_edge_case(conn, frame_id, file_name, timestamp, edge_cases, motion_data):
    """Store detected edge cases to database."""
    try:
        cursor = conn.cursor()
        
        for edge_case in edge_cases:
            cursor.execute('''
                INSERT INTO edge_cases 
                (frame_id, file_name, timestamp, edge_case_type, severity, 
                 speed_min, speed_max, accel_x_min, accel_y_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                frame_id,
                file_name,
                timestamp,
                edge_case['type'],
                edge_case['severity'],
                motion_data['speed_min'],
                motion_data['speed_max'],
                motion_data['accel_x_min'],
                motion_data['accel_y_max']
            ))
        
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"✗ Database write error: {e}")
        print(f"  Retrying write operation...")
        try:
            conn.rollback()
            cursor = conn.cursor()
            for edge_case in edge_cases:
                cursor.execute('''
                    INSERT INTO edge_cases 
                    (frame_id, file_name, timestamp, edge_case_type, severity, 
                     speed_min, speed_max, accel_x_min, accel_y_max)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    frame_id,
                    file_name,
                    timestamp,
                    edge_case['type'],
                    edge_case['severity'],
                    motion_data['speed_min'],
                    motion_data['speed_max'],
                    motion_data['accel_x_min'],
                    motion_data['accel_y_max']
                ))
            conn.commit()
            print(f"  ✓ Retry successful")
        except Exception as retry_err:
            print(f"  ✗ Retry failed: {retry_err}")

# ============================================================================
# THRESHOLD MANAGEMENT
# ============================================================================
def load_or_calculate_thresholds(filename, force_recalculate=False):
    """
    Load thresholds from file or calculate adaptively from dataset.
    
    Args:
        filename: Path to tfrecord file
        force_recalculate: Force recalculation even if file exists
    
    Returns:
        Dictionary of thresholds
    """
    # Try to load from file
    if os.path.exists(THRESHOLD_FILE) and not force_recalculate:
        try:
            with open(THRESHOLD_FILE, 'r') as f:
                thresholds = json.load(f)
            print(f"✓ Loaded saved thresholds from {THRESHOLD_FILE}")
            return thresholds
        except Exception as e:
            print(f"⚠ Could not load thresholds: {e}")
    
    # Calculate adaptively from current file
    print("Calculating adaptive thresholds from dataset...")
    all_accel_x = []
    all_accel_y = []
    all_jerk = []
    
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    for bytes_example in dataset.as_numpy_iterator():
        data = wod_e2ed_pb2.E2EDFrame()
        data.ParseFromString(bytes_example)
        accel_x = np.array(data.past_states.accel_x)
        accel_y = np.array(data.past_states.accel_y)
        all_accel_x.extend(accel_x)
        all_accel_y.extend(accel_y)
        all_jerk.extend(np.abs(np.diff(accel_x)))
    
    # Calculate percentiles (top 5% are anomalies)
    thresholds = {
        'hard_brake': float(np.percentile(all_accel_x, 5)),      # Lowest 5%
        'lateral': float(np.percentile(np.abs(all_accel_y), 95)),  # Highest 5%
        'jerk': float(np.percentile(all_jerk, 95))                 # Highest 5%
    }
    
    # Save for future use
    try:
        with open(THRESHOLD_FILE, 'w') as f:
            json.dump(thresholds, f, indent=2)
        print(f"✓ Saved thresholds to {THRESHOLD_FILE}")
    except Exception as e:
        print(f"⚠ Could not save thresholds: {e}")
    
    return thresholds

# ============================================================================
# LOAD DATA
# ============================================================================
if len(sys.argv) < 2:
    print("Error: Please provide a filename as an argument.")
    sys.exit(1)

filename = sys.argv[1]
print(f"--- Processing file: {filename} ---")

try:
    dataset = tf.data.TFRecordDataset(filename, compression_type='')
    data_iter = dataset.as_numpy_iterator()
    print("✓ Successfully loaded dataset")
except Exception as e:
    print(f"✗ Error processing file {filename}: {e}")
    sys.exit(1)

# Initialize database
db_conn = init_database(DB_PATH)
print(f"✓ Database initialized: {DB_PATH}")

# ============================================================================
# LOAD THRESHOLDS (HYBRID APPROACH)
# ============================================================================
print("\n--- Threshold Management ---")
THRESHOLDS = load_or_calculate_thresholds(filename, force_recalculate=False)

print(f"\n✓ Active Thresholds:")
print(f"  Hard brake: {THRESHOLDS['hard_brake']:.4f} m/s²")
print(f"  Lateral: {THRESHOLDS['lateral']:.4f} m/s²")
print(f"  Jerk: {THRESHOLDS['jerk']:.4f} m/s³\n")

# Reload dataset for processing
dataset = tf.data.TFRecordDataset(filename, compression_type='')
data_iter = dataset.as_numpy_iterator()

# ============================================================================
# PROCESS DATA
# ============================================================================
frame_count = 0
edge_case_count = 0

for bytes_example in data_iter:
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(bytes_example)
    frame_count += 1
    
    print(f"\n--- Frame {frame_count} ---")
    
    # Extract PAST motion data for edge case detection
    vel_x = np.array(data.past_states.vel_x)
    vel_y = np.array(data.past_states.vel_y)
    accel_x = np.array(data.past_states.accel_x)
    accel_y = np.array(data.past_states.accel_y)
    
    # Calculate speed from velocity
    speed = np.sqrt(vel_x**2 + vel_y**2)
    
    print(f"Speed range: {speed.min():.2f} - {speed.max():.2f} m/s")
    print(f"Max acceleration: {np.max(np.abs([accel_x, accel_y])):.2f} m/s²")
    
    # Detect edge cases using HYBRID thresholds
    detected_edge_cases = []
    
    hard_brake = np.min(accel_x) < THRESHOLDS['hard_brake']
    high_lateral_accel = np.max(np.abs(accel_y)) > THRESHOLDS['lateral']
    high_jerk = np.max(np.abs(np.diff(accel_x))) > THRESHOLDS['jerk']
    
    if hard_brake:
        severity = abs(np.min(accel_x))
        detected_edge_cases.append({'type': 'hard_brake', 'severity': severity})
        print(f"⚠ HARD BRAKE DETECTED: {severity:.4f} m/s²")
        edge_case_count += 1
    
    if high_lateral_accel:
        severity = np.max(np.abs(accel_y))
        detected_edge_cases.append({'type': 'evasive_maneuver', 'severity': severity})
        print(f"⚠ EVASIVE MANEUVER DETECTED: {severity:.4f} m/s²")
        edge_case_count += 1
    
    if high_jerk:
        severity = np.max(np.abs(np.diff(accel_x)))
        detected_edge_cases.append({'type': 'high_jerk', 'severity': severity})
        print(f"⚠ HIGH JERK DETECTED: {severity:.4f} m/s³")
        edge_case_count += 1
    
    # Store edge cases to database
    if detected_edge_cases:
        motion_data = {
            'speed_min': speed.min(),
            'speed_max': speed.max(),
            'accel_x_min': accel_x.min(),
            'accel_y_max': np.max(np.abs(accel_y))
        }
        store_edge_case(
            db_conn,
            frame_id=frame_count,
            file_name=os.path.basename(filename),
            timestamp=data.frame.timestamp_micros,
            edge_cases=detected_edge_cases,
            motion_data=motion_data
        )
        print(f"✓ Stored {len(detected_edge_cases)} edge case(s) to database")

# ============================================================================
# SUMMARY & CLEANUP
# ============================================================================
print(f"\n--- Processing complete: {frame_count} frames processed ---")
print(f"--- Total edge cases detected: {edge_case_count} ---")

# Query database summary
try:
    cursor = db_conn.cursor()
    cursor.execute('SELECT edge_case_type, COUNT(*) FROM edge_cases GROUP BY edge_case_type')
    results = cursor.fetchall()
    print("\n--- Edge Case Summary ---")
    for edge_type, count in results:
        print(f"  {edge_type}: {count}")
except Exception as e:
    print(f"⚠ Could not retrieve summary: {e}")

db_conn.close()
print(f"✓ Database closed: {DB_PATH}")