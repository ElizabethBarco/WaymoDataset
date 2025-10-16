import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = os.getenv('RESULTS_DIR', './waymo_dataset/results')
DB_PATH = os.path.join(RESULTS_DIR, 'edge_cases.db')

# ============================================================================
# VERIFY DATABASE EXISTS
# ============================================================================
if not os.path.exists(DB_PATH):
    print(f"✗ Database not found at {DB_PATH}")
    print(f"  Please run load_dataset.py first to populate the database")
    exit(1)

try:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM edge_cases", conn)
    conn.close()
    
    if len(df) == 0:
        print("⚠ Database is empty - no edge cases found yet")
        exit(1)
    
    # ============================================================================
    # ANALYSIS
    # ============================================================================
    print(f"\n✓ Total edge cases: {len(df)}")
    print(f"\nBreakdown by type:")
    print(df['edge_case_type'].value_counts())
    
    print(f"\nSeverity statistics:")
    print(df.groupby('edge_case_type')['severity'].describe())
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    # Ensure output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(column='severity', by='edge_case_type', ax=ax)
    plt.title('Edge Case Severity Distribution')
    plt.suptitle('')  # Remove default title
    plt.tight_layout()
    
    output_path = os.path.join(RESULTS_DIR, 'edge_case_analysis.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Analysis plot saved to {output_path}")
    
    # ============================================================================
    # EXPORT TO CSV
    # ============================================================================
    csv_path = os.path.join(RESULTS_DIR, 'edge_cases.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Data exported to {csv_path}")
    
    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print(f"\n--- Summary Statistics ---")
    print(f"Total files processed: {df['file_name'].nunique()}")
    print(f"Total edge cases: {len(df)}")
    
    for edge_type in sorted(df['edge_case_type'].unique()):
        subset = df[df['edge_case_type'] == edge_type]
        print(f"\n{edge_type.upper()}:")
        print(f"  Count: {len(subset)}")
        print(f"  Min severity: {subset['severity'].min():.4f}")
        print(f"  Max severity: {subset['severity'].max():.4f}")
        print(f"  Mean severity: {subset['severity'].mean():.4f}")
        print(f"  Std severity: {subset['severity'].std():.4f}")
    
    print(f"\n✓ Analysis complete!")
    
except sqlite3.OperationalError as e:
    print(f"✗ Database error: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)