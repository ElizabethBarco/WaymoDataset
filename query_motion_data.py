"""
Query and analyze complete motion dataset stored in SQLite.
Demonstrates the power of storing ALL frames vs just edge cases.
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = os.getenv('RESULTS_DIR', './waymo_dataset/results')
DB_PATH = os.path.join(RESULTS_DIR, 'edge_cases.db')

# ============================================================================
# CONNECTION
# ============================================================================
def get_connection(db_path):
    """Get database connection."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        return conn
    except Exception as e:
        print(f"✗ Could not connect to database: {e}")
        return None

# ============================================================================
# EXAMPLE QUERIES
# ============================================================================
def query_example_1(conn):
    """Find extreme 1% acceleration events."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Find extreme 1% deceleration events (hardest braking)")
    print("="*80)
    
    query = '''
        SELECT 
            frame_id, file_name, intent, accel_x_min, speed_max, 
            (SELECT COUNT(*) FROM frames WHERE accel_x_min < f.accel_x_min) as rank
        FROM frames f
        ORDER BY accel_x_min ASC
        LIMIT 10
    '''
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    print(f"\nThese {len(df)} frames represent the hardest braking events in the dataset")


def query_example_2(conn):
    """Find unusual lateral acceleration during straight driving."""
    print("\n" + "="*80)
    print("EXAMPLE 2: High lateral acceleration WHILE intent=GO_STRAIGHT (suspicious!)")
    print("="*80)
    
    query = '''
        SELECT 
            frame_id, file_name, intent, accel_y_max, speed_max,
            accel_x_min, accel_x_max
        FROM frames
        WHERE intent = 'GO_STRAIGHT' AND accel_y_max > 0.3
        ORDER BY accel_y_max DESC
        LIMIT 10
    '''
    
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        print(df.to_string(index=False))
        print(f"\n⚠ Found {len(df)} suspicious straight-driving maneuvers with high lateral accel!")
    else:
        print("No suspicious maneuvers found")


def query_example_3(conn):
    """Calculate dynamic thresholds from full dataset."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Dynamic threshold calculation (full dataset statistics)")
    print("="*80)
    
    query = '''
        SELECT 
            COUNT(*) as total_frames,
            ROUND(AVG(accel_x_min), 4) as mean_accel_x_min,
            ROUND(MIN(accel_x_min), 4) as min_accel_x,
            ROUND(MAX(accel_x_min), 4) as max_accel_x
        FROM frames
    '''
    
    stats = pd.read_sql_query(query, conn)
    print(stats.to_string(index=False))
    
    # Calculate 1%, 5%, 25% percentiles for intelligent thresholding
    print("\n--- Percentile-based thresholds for accel_x_min ---")
    query_percentiles = '''
        WITH ranked AS (
            SELECT 
                accel_x_min,
                ROW_NUMBER() OVER (ORDER BY accel_x_min) as row_num,
                COUNT(*) OVER () as total
            FROM frames
        )
        SELECT 
            ROUND(AVG(accel_x_min), 4) as p01_accel_x
        FROM ranked
        WHERE row_num <= CEIL(total * 0.01)
        UNION ALL
        SELECT 
            ROUND(AVG(accel_x_min), 4)
        FROM ranked
        WHERE row_num <= CEIL(total * 0.05)
        UNION ALL
        SELECT 
            ROUND(AVG(accel_x_min), 4)
        FROM ranked
        WHERE row_num <= CEIL(total * 0.25)
    '''
    
    percentiles = pd.read_sql_query(query_percentiles, conn)
    print(f"  1st percentile (extreme 1%): {percentiles.iloc[0, 0]:.4f} m/s²")
    print(f"  5th percentile (extreme 5%): {percentiles.iloc[1, 0]:.4f} m/s²")
    print(f"  25th percentile (extreme 25%): {percentiles.iloc[2, 0]:.4f} m/s²")


def query_example_4(conn):
    """Compare intent-based motion profiles."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Compare motion profiles by driving intent")
    print("="*80)
    
    query = '''
        SELECT 
            intent,
            COUNT(*) as frame_count,
            ROUND(AVG(speed_mean), 2) as avg_speed,
            ROUND(AVG(ABS(accel_x_min)), 2) as avg_decel,
            ROUND(AVG(ABS(accel_y_max)), 2) as avg_lateral,
            ROUND(AVG(jerk_x_max), 3) as avg_jerk,
            ROUND(MAX(accel_y_max), 2) as max_lateral,
            ROUND(MIN(accel_x_min), 2) as min_accel_x
        FROM frames
        GROUP BY intent
        ORDER BY frame_count DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    print("\nKey insight: Compare avg_decel/avg_lateral across intents to calibrate thresholds")


def query_example_5(conn):
    """Find frames flagged as edge cases and analyze them."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Analyze flagged edge cases in context")
    print("="*80)
    
    query = '''
        SELECT 
            ec.edge_case_type,
            COUNT(*) as count,
            ROUND(AVG(ec.severity), 4) as avg_severity,
            ROUND(MIN(ec.severity), 4) as min_severity,
            ROUND(MAX(ec.severity), 4) as max_severity,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM frames), 2) as percent_of_total
        FROM edge_cases ec
        GROUP BY ec.edge_case_type
        ORDER BY count DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No edge cases flagged yet")


def query_example_6(conn):
    """Speed distribution analysis - when do edge cases happen?"""
    print("\n" + "="*80)
    print("EXAMPLE 6: When do edge cases occur? Speed distribution analysis")
    print("="*80)
    
    query = '''
        SELECT 
            ec.edge_case_type,
            CASE 
                WHEN f.speed_max < 3 THEN 'stopped/parking'
                WHEN f.speed_max < 10 THEN 'slow (0-25 mph)'
                WHEN f.speed_max < 20 THEN 'medium (25-50 mph)'
                WHEN f.speed_max < 30 THEN 'fast (50-75 mph)'
                ELSE 'very_fast (75+ mph)'
            END as speed_category,
            COUNT(*) as edge_case_count
        FROM edge_cases ec
        JOIN frames f ON ec.frame_id = f.frame_id
        GROUP BY ec.edge_case_type, speed_category
        ORDER BY ec.edge_case_type, edge_case_count DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    if len(df) > 0:
        print(df.to_string(index=False))
    else:
        print("No edge cases to analyze")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    print(f"Connecting to database: {DB_PATH}")
    conn = get_connection(DB_PATH)
    
    if conn:
        try:
            # Run example queries
            query_example_1(conn)
            query_example_2(conn)
            query_example_3(conn)
            query_example_4(conn)
            query_example_5(conn)
            query_example_6(conn)
            
            print("\n" + "="*80)
            print("✓ Query analysis complete")
            print("="*80)
            
        except Exception as e:
            print(f"✗ Error during analysis: {e}")
        finally:
            conn.close()
    else:
        print("✗ Could not connect to database")
