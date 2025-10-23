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
    """Initialize SQLite database for comprehensive motion data storage."""
    try:
        # Use check_same_thread=False for container compatibility
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        conn.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging for better concurrency
        cursor = conn.cursor()
        
        # Main frames table - stores ALL frame motion data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                file_name TEXT,
                timestamp BIGINT,
                intent TEXT,
                speed_min REAL,
                speed_max REAL,
                speed_mean REAL,
                accel_x_min REAL,
                accel_x_max REAL,
                accel_y_min REAL,
                accel_y_max REAL,
                jerk_x_max REAL,
                jerk_y_max REAL,
                panorama_thumbnail BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Edge cases table - stores flagged anomalies (references frames table)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edge_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frame_id INTEGER,
                file_name TEXT,
                timestamp BIGINT,
                edge_case_type TEXT,
                severity REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(frame_id) REFERENCES frames(frame_id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_intent ON frames(intent)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_accel_x_min ON frames(accel_x_min)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_frames_accel_y_max ON frames(accel_y_max)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_edge_cases_type ON edge_cases(edge_case_type)')
        
        conn.commit()
        return conn
    except Exception as e:
        print(f"✗ Database initialization error: {e}")
        raise

def store_frame_data(conn, frame_id, file_name, timestamp, motion_data, intent, panorama_thumbnail_bytes=None):
    """Store complete motion profile for every frame."""
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO frames 
            (frame_id, file_name, timestamp, intent, speed_min, speed_max, speed_mean, 
             accel_x_min, accel_x_max, accel_y_min, accel_y_max, jerk_x_max, jerk_y_max, panorama_thumbnail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            frame_id,
            file_name,
            timestamp,
            intent,
            motion_data['speed_min'],
            motion_data['speed_max'],
            motion_data['speed_mean'],
            motion_data['accel_x_min'],
            motion_data['accel_x_max'],
            motion_data['accel_y_min'],
            motion_data['accel_y_max'],
            motion_data['jerk_x_max'],
            motion_data['jerk_y_max'],
            panorama_thumbnail_bytes
        ))
        
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"✗ Database write error: {e}")
        try:
            conn.rollback()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO frames 
                (frame_id, file_name, timestamp, intent, speed_min, speed_max, speed_mean, 
                 accel_x_min, accel_x_max, accel_y_min, accel_y_max, jerk_x_max, jerk_y_max, panorama_thumbnail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                frame_id,
                file_name,
                timestamp,
                intent,
                motion_data['speed_min'],
                motion_data['speed_max'],
                motion_data['speed_mean'],
                motion_data['accel_x_min'],
                motion_data['accel_x_max'],
                motion_data['accel_y_min'],
                motion_data['accel_y_max'],
                motion_data['jerk_x_max'],
                motion_data['jerk_y_max'],
                panorama_thumbnail_bytes
            ))
            conn.commit()
        except Exception as retry_err:
            print(f"  ✗ Retry failed: {retry_err}")

def store_edge_case(conn, frame_id, file_name, timestamp, edge_case_type, severity, reason):
    """Store flagged edge case to database."""
    try:
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO edge_cases 
            (frame_id, file_name, timestamp, edge_case_type, severity, reason)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            frame_id,
            file_name,
            timestamp,
            edge_case_type,
            severity,
            reason
        ))
        
        conn.commit()
    except sqlite3.OperationalError as e:
        print(f"✗ Database write error: {e}")
        try:
            conn.rollback()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO edge_cases 
                (frame_id, file_name, timestamp, edge_case_type, severity, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                frame_id,
                file_name,
                timestamp,
                edge_case_type,
                severity,
                reason
            ))
            conn.commit()
        except Exception as retry_err:
            print(f"  ✗ Retry failed: {retry_err}")

# ============================================================================
# IMAGE STITCHING & THUMBNAIL GENERATION
# ============================================================================
# TEMPORARILY DISABLED - Image processing functions removed for now
# To re-enable: restore stitch_panorama(), create_thumbnail(), compress_image_to_bytes()

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
    
    # Extract PAST motion data
    vel_x = np.array(data.past_states.vel_x)
    vel_y = np.array(data.past_states.vel_y)
    accel_x = np.array(data.past_states.accel_x)
    accel_y = np.array(data.past_states.accel_y)
    
    # Calculate motion metrics
    speed = np.sqrt(vel_x**2 + vel_y**2)
    jerk_x = np.abs(np.diff(accel_x)) if len(accel_x) > 1 else np.array([0])
    jerk_y = np.abs(np.diff(accel_y)) if len(accel_y) > 1 else np.array([0])
    
    # Get intent
    intent_map = {
        0: 'UNKNOWN',
        1: 'GO_STRAIGHT',
        2: 'GO_LEFT',
        3: 'GO_RIGHT'
    }
    intent = intent_map.get(data.intent, 'UNKNOWN')
    
    # Compile motion data
    motion_data = {
        'speed_min': float(speed.min()),
        'speed_max': float(speed.max()),
        'speed_mean': float(speed.mean()),
        'accel_x_min': float(accel_x.min()),
        'accel_x_max': float(accel_x.max()),
        'accel_y_min': float(accel_y.min()),
        'accel_y_max': float(accel_y.max()),
        'jerk_x_max': float(jerk_x.max()),
        'jerk_y_max': float(jerk_y.max())
    }
    
    print(f"Speed range: {motion_data['speed_min']:.2f} - {motion_data['speed_max']:.2f} m/s")
    print(f"Accel X: {motion_data['accel_x_min']:.2f} to {motion_data['accel_x_max']:.2f} m/s²")
    print(f"Accel Y: {motion_data['accel_y_min']:.2f} to {motion_data['accel_y_max']:.2f} m/s²")
    print(f"Intent: {intent}")
    
    # Skip image processing for now - store as NULL
    # TODO: Re-enable when image storage is needed
    panorama_thumbnail_bytes = None
    
    # STORE ALL FRAME DATA (regardless of whether it's an edge case)
    store_frame_data(
        db_conn,
        frame_id=frame_count,
        file_name=os.path.basename(filename),
        timestamp=data.frame.timestamp_micros,
        motion_data=motion_data,
        intent=intent,
        panorama_thumbnail_bytes=panorama_thumbnail_bytes
    )
    
    # DETECT AND FLAG EDGE CASES
    hard_brake = motion_data['accel_x_min'] < THRESHOLDS['hard_brake']
    high_lateral_accel = motion_data['accel_y_max'] > THRESHOLDS['lateral']
    high_jerk = motion_data['jerk_x_max'] > THRESHOLDS['jerk']
    
    if hard_brake:
        severity = abs(motion_data['accel_x_min'])
        reason = f"accel_x={motion_data['accel_x_min']:.3f} < threshold {THRESHOLDS['hard_brake']:.3f}"
        store_edge_case(db_conn, frame_count, os.path.basename(filename), 
                       data.frame.timestamp_micros, 'hard_brake', severity, reason)
        print(f"⚠ HARD BRAKE FLAGGED: {severity:.4f} m/s²")
        edge_case_count += 1
    
    if high_lateral_accel:
        severity = motion_data['accel_y_max']
        reason = f"accel_y={motion_data['accel_y_max']:.3f} > threshold {THRESHOLDS['lateral']:.3f}"
        store_edge_case(db_conn, frame_count, os.path.basename(filename), 
                       data.frame.timestamp_micros, 'evasive_maneuver', severity, reason)
        print(f"⚠ EVASIVE MANEUVER FLAGGED: {severity:.4f} m/s²")
        edge_case_count += 1
    
    if high_jerk:
        severity = motion_data['jerk_x_max']
        reason = f"jerk_x={motion_data['jerk_x_max']:.3f} > threshold {THRESHOLDS['jerk']:.3f}"
        store_edge_case(db_conn, frame_count, os.path.basename(filename), 
                       data.frame.timestamp_micros, 'high_jerk', severity, reason)
        print(f"⚠ HIGH JERK FLAGGED: {severity:.4f} m/s³")
        edge_case_count += 1

# ============================================================================
# SUMMARY & CLEANUP
# ============================================================================
print(f"\n--- Processing complete: {frame_count} frames processed ---")
print(f"--- Total edge cases flagged: {edge_case_count} ---")

# Query database summary
try:
    cursor = db_conn.cursor()
    
    print("\n--- Frames Table Summary ---")
    cursor.execute('SELECT COUNT(*) FROM frames')
    total_frames = cursor.fetchone()[0]
    print(f"Total frames stored: {total_frames}")
    
    cursor.execute('SELECT intent, COUNT(*) FROM frames GROUP BY intent ORDER BY COUNT(*) DESC')
    for intent, count in cursor.fetchall():
        print(f"  {intent}: {count}")
    
    print("\n--- Edge Cases Table Summary ---")
    cursor.execute('SELECT edge_case_type, COUNT(*) FROM edge_cases GROUP BY edge_case_type')
    edge_results = cursor.fetchall()
    if edge_results:
        for edge_type, count in edge_results:
            print(f"  {edge_type}: {count}")
    else:
        print("  No edge cases flagged")
        
    # Calculate statistics for thresholding
    print("\n--- Motion Statistics (for future recalibration) ---")
    cursor.execute('''
        SELECT 
            MIN(accel_x_min) as min_accel_x,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY accel_x_min) as p05_accel_x,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY accel_x_min) as p25_accel_x,
            MAX(speed_max) as max_speed,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY accel_y_max) as p95_accel_y,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY jerk_x_max) as p95_jerk_x
        FROM frames
    ''')
    stats = cursor.fetchone()
    if stats:
        print(f"  Min accel X: {stats[0]:.4f} m/s²")
        print(f"  5th percentile accel X: {stats[1]:.4f} m/s²")
        print(f"  25th percentile accel X: {stats[2]:.4f} m/s²")
        print(f"  Max speed: {stats[3]:.4f} m/s")
        print(f"  95th percentile accel Y: {stats[4]:.4f} m/s²")
        print(f"  95th percentile jerk X: {stats[5]:.4f} m/s³")
    
except Exception as e:
    print(f"⚠ Could not retrieve summary: {e}")

db_conn.close()
print(f"\n✓ Database closed: {DB_PATH}")