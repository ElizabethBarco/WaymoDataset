"""
Test script to view panorama thumbnails extracted from the database.
Retrieves thumbnails as JPEG bytes and displays/saves them for verification.
"""

import sqlite3
import os
import base64
from pathlib import Path
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = os.getenv('RESULTS_DIR', './waymo_dataset/results')
DB_PATH = os.path.join(RESULTS_DIR, 'edge_cases.db')
OUTPUT_DIR = os.path.join(RESULTS_DIR, 'thumbnail_test_exports')

# ============================================================================
# SETUP
# ============================================================================
def setup_output_dir():
    """Create output directory for test exports."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

def check_database():
    """Verify database exists and has frames with thumbnails."""
    if not os.path.exists(DB_PATH):
        print(f"✗ Database not found at {DB_PATH}")
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if frames table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frames'")
        if not cursor.fetchone():
            print("✗ 'frames' table not found in database")
            return False
        
        # Check how many frames have thumbnails
        cursor.execute("SELECT COUNT(*) as total, COUNT(panorama_thumbnail) as with_thumbnails FROM frames")
        total, with_thumbnails = cursor.fetchone()
        
        print(f"✓ Database check:")
        print(f"  Total frames: {total}")
        print(f"  Frames with thumbnails: {with_thumbnails}")
        
        if with_thumbnails == 0:
            print("⚠ No thumbnails found in database yet")
            return total > 0  # Return True if frames exist (thumbnails might be NULL)
        
        conn.close()
        return True
    
    except Exception as e:
        print(f"✗ Error checking database: {e}")
        return False

# ============================================================================
# THUMBNAIL EXTRACTION & VIEWING
# ============================================================================
def get_frames_with_thumbnails(limit=10):
    """
    Retrieve frames with thumbnail data from database.
    
    Args:
        limit: Maximum number of frames to retrieve
    
    Returns:
        List of frame dictionaries with thumbnail data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT 
            id,
            frame_id,
            file_name,
            intent,
            speed_max,
            accel_x_min,
            accel_y_max,
            panorama_thumbnail
        FROM frames
        WHERE panorama_thumbnail IS NOT NULL
        LIMIT ?
        '''
        
        cursor.execute(query, (limit,))
        frames = cursor.fetchall()
        conn.close()
        
        return [dict(frame) for frame in frames]
    
    except Exception as e:
        print(f"✗ Error retrieving frames: {e}")
        return []

def save_thumbnail_as_file(thumbnail_bytes, output_path):
    """
    Save thumbnail bytes to a JPEG file.
    
    Args:
        thumbnail_bytes: Raw JPEG bytes from database
        output_path: Path where to save the file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'wb') as f:
            f.write(thumbnail_bytes)
        return True
    except Exception as e:
        print(f"  ✗ Error saving file: {e}")
        return False

def create_html_viewer(frames_data):
    """
    Create an HTML file to view all thumbnails in a browser.
    
    Args:
        frames_data: List of frame dictionaries with thumbnail data
    """
    try:
        html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Panorama Thumbnail Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1a1a1a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 30px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .frame-card {
            background-color: #2a2a2a;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .frame-card h3 {
            margin-top: 0;
            color: #4CAF50;
            border-bottom: 1px solid #4CAF50;
            padding-bottom: 10px;
        }
        .frame-image {
            width: 100%;
            height: auto;
            border-radius: 4px;
            margin: 15px 0;
            border: 1px solid #444;
        }
        .frame-metadata {
            font-size: 12px;
            color: #cccccc;
            line-height: 1.6;
        }
        .metadata-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #444;
        }
        .metadata-label {
            font-weight: bold;
            color: #4CAF50;
            width: 150px;
        }
        .metadata-value {
            color: #ffffff;
            flex: 1;
            word-break: break-all;
        }
        .status {
            text-align: center;
            padding: 20px;
            margin: 20px;
            background-color: #2a2a2a;
            border-radius: 8px;
            border: 2px solid #4CAF50;
        }
        .status.success {
            color: #4CAF50;
        }
        .status.warning {
            color: #FFC107;
        }
    </style>
</head>
<body>
    <h1>🚗 Waymo Panorama Thumbnail Viewer</h1>
    <div class="status success">
        Successfully loaded {0} frames with thumbnails from database
    </div>
    <div class="container">
'''
        
        # Add each frame as a card
        for frame in frames_data:
            thumbnail_b64 = base64.b64encode(frame['thumbnail_bytes']).decode('utf-8')
            
            html_content += f'''        <div class="frame-card">
            <h3>Frame {frame['frame_id']}</h3>
            <img src="data:image/jpeg;base64,{thumbnail_b64}" alt="Panorama Thumbnail" class="frame-image">
            <div class="frame-metadata">
                <div class="metadata-row">
                    <span class="metadata-label">Frame ID:</span>
                    <span class="metadata-value">{frame['frame_id']}</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">File:</span>
                    <span class="metadata-value">{frame['file_name']}</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Intent:</span>
                    <span class="metadata-value">{frame['intent']}</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Speed Max:</span>
                    <span class="metadata-value">{frame['speed_max']:.2f} m/s</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Accel X Min:</span>
                    <span class="metadata-value">{frame['accel_x_min']:.3f} m/s²</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Accel Y Max:</span>
                    <span class="metadata-value">{frame['accel_y_max']:.3f} m/s²</span>
                </div>
                <div class="metadata-row">
                    <span class="metadata-label">Thumbnail Size:</span>
                    <span class="metadata-value">{len(frame['thumbnail_bytes']) / 1024:.1f} KB</span>
                </div>
            </div>
        </div>
'''
        
        html_content += '''    </div>
</body>
</html>
'''
        
        html_path = os.path.join(OUTPUT_DIR, 'thumbnail_viewer.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Created HTML viewer: {html_path}")
        return html_path
    
    except Exception as e:
        print(f"✗ Error creating HTML viewer: {e}")
        return None

# ============================================================================
# MAIN TEST ROUTINE
# ============================================================================
def main():
    print("=" * 70)
    print("🎬 Waymo Panorama Thumbnail Database Test")
    print("=" * 70)
    
    setup_output_dir()
    
    print("\n--- Checking Database ---")
    if not check_database():
        print("✗ Database check failed")
        return False
    
    print("\n--- Retrieving Frames with Thumbnails ---")
    frames = get_frames_with_thumbnails(limit=10)
    
    if not frames:
        print("⚠ No frames with thumbnails found")
        print("  This is normal if load_dataset.py hasn't been run yet")
        print("  Or if frames were added before thumbnail support was added")
        return False
    
    print(f"✓ Retrieved {len(frames)} frames with thumbnails")
    
    print("\n--- Exporting Thumbnail Files ---")
    for i, frame in enumerate(frames, 1):
        thumbnail_bytes = frame['panorama_thumbnail']
        
        if thumbnail_bytes is None:
            print(f"  Frame {i}: Thumbnail is NULL (skipping)")
            continue
        
        filename = f"frame_{frame['frame_id']}_thumbnail.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        if save_thumbnail_as_file(thumbnail_bytes, filepath):
            size_kb = len(thumbnail_bytes) / 1024
            print(f"  ✓ Frame {frame['frame_id']}: {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ Frame {frame['frame_id']}: Failed to save")
    
    print("\n--- Creating HTML Viewer ---")
    # Prepare data for HTML viewer (include thumbnail bytes)
    frames_for_html = [
        {
            'frame_id': f['frame_id'],
            'file_name': f['file_name'],
            'intent': f['intent'],
            'speed_max': f['speed_max'],
            'accel_x_min': f['accel_x_min'],
            'accel_y_max': f['accel_y_max'],
            'thumbnail_bytes': f['panorama_thumbnail']
        }
        for f in frames if f['panorama_thumbnail'] is not None
    ]
    
    if frames_for_html:
        html_path = create_html_viewer(frames_for_html)
        if html_path:
            print(f"  Open in browser: file://{os.path.abspath(html_path)}")
    
    print("\n" + "=" * 70)
    print("✓ Thumbnail test complete!")
    print(f"✓ All exports saved to: {OUTPUT_DIR}")
    print("=" * 70)
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
