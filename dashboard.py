import sqlite3
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html, callback, Input, Output
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
RESULTS_DIR = os.getenv('RESULTS_DIR', './waymo_dataset/results')
DB_PATH = os.path.join(RESULTS_DIR, 'edge_cases.db')

# ============================================================================
# DATABASE SETUP
# ============================================================================
if not os.path.exists(DB_PATH):
    print(f"✗ Database not found at {DB_PATH}")
    print(f"  Please run load_dataset.py first to populate the database")
    exit(1)

def get_edge_cases_data():
    """Load edge cases from database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM edge_cases", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"✗ Error loading database: {e}")
        exit(1)

# Load data once
df = get_edge_cases_data()

if len(df) == 0:
    print("⚠ Database is empty - no edge cases found")
    exit(1)

print(f"✓ Loaded {len(df)} edge cases from database")

# ============================================================================
# INITIALIZE DASH APP
# ============================================================================
app = Dash(__name__)

# ============================================================================
# APP LAYOUT
# ============================================================================
app.layout = html.Div([
    html.Div([
        html.H1("🚗 Waymo Edge Case Detection Dashboard", style={'textAlign': 'center', 'marginBottom': 30}),
    ], style={'backgroundColor': '#1f77b4', 'color': 'white', 'padding': '20px', 'borderRadius': '10px'}),
    
    html.Div([
        # Summary Statistics Row
        html.Div([
            html.Div([
                html.H3(f"{len(df)}", style={'color': '#d62728', 'margin': 0}),
                html.P("Total Edge Cases", style={'margin': 0})
            ], style={
                'backgroundColor': '#f0f0f0',
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'flex': 1,
                'margin': '10px'
            }),
            
            html.Div([
                html.H3(f"{df['file_name'].nunique()}", style={'color': '#2ca02c', 'margin': 0}),
                html.P("Files Processed", style={'margin': 0})
            ], style={
                'backgroundColor': '#f0f0f0',
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'flex': 1,
                'margin': '10px'
            }),
            
            html.Div([
                html.H3(f"{df['edge_case_type'].nunique()}", style={'color': '#ff7f0e', 'margin': 0}),
                html.P("Edge Case Types", style={'margin': 0})
            ], style={
                'backgroundColor': '#f0f0f0',
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'flex': 1,
                'margin': '10px'
            }),
            
            html.Div([
                html.H3(f"{df['severity'].max():.2f}", style={'color': '#9467bd', 'margin': 0}),
                html.P("Max Severity", style={'margin': 0})
            ], style={
                'backgroundColor': '#f0f0f0',
                'padding': '20px',
                'borderRadius': '8px',
                'textAlign': 'center',
                'flex': 1,
                'margin': '10px'
            }),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
        
        # Filters Row
        html.Div([
            html.Div([
                html.Label("Filter by Edge Case Type:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='edge-case-filter',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        *[{'label': f, 'value': f} for f in sorted(df['edge_case_type'].unique())]
                    ],
                    value='all',
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.Label("Filter by File:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='file-filter',
                    options=[
                        {'label': 'All Files', 'value': 'all'},
                        *[{'label': f, 'value': f} for f in sorted(df['file_name'].unique())]
                    ],
                    value='all',
                    style={'width': '100%'}
                ),
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
            
            html.Div([
                html.Label("Severity Range:", style={'fontWeight': 'bold'}),
                dcc.RangeSlider(
                    id='severity-slider',
                    min=df['severity'].min(),
                    max=df['severity'].max(),
                    value=[df['severity'].min(), df['severity'].max()],
                    marks={i: f'{i:.1f}' for i in [df['severity'].min(), df['severity'].max()]},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '35%', 'display': 'inline-block'}),
        ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
        
        # Charts Row 1
        html.Div([
            html.Div([
                dcc.Graph(id='edge-case-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='severity-distribution')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
        ]),
        
        # Charts Row 2
        html.Div([
            html.Div([
                dcc.Graph(id='severity-by-type')
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
            
            html.Div([
                dcc.Graph(id='edge-cases-by-file')
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '2%'}),
        ]),
        
        # Data Table
        html.Div([
            html.H3("Edge Cases Details", style={'marginTop': '40px', 'marginBottom': '20px'}),
            html.Div(id='data-table')
        ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'borderRadius': '8px'}),
        
    ], style={'padding': '20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff'})

# ============================================================================
# CALLBACKS
# ============================================================================
@callback(
    [Output('edge-case-distribution', 'figure'),
     Output('severity-distribution', 'figure'),
     Output('severity-by-type', 'figure'),
     Output('edge-cases-by-file', 'figure'),
     Output('data-table', 'children')],
    [Input('edge-case-filter', 'value'),
     Input('file-filter', 'value'),
     Input('severity-slider', 'value')]
)
def update_charts(edge_case_type, file_name, severity_range):
    """Update all charts based on filters."""
    
    # Apply filters
    filtered_df = df.copy()
    
    if edge_case_type != 'all':
        filtered_df = filtered_df[filtered_df['edge_case_type'] == edge_case_type]
    
    if file_name != 'all':
        filtered_df = filtered_df[filtered_df['file_name'] == file_name]
    
    filtered_df = filtered_df[
        (filtered_df['severity'] >= severity_range[0]) & 
        (filtered_df['severity'] <= severity_range[1])
    ]
    
    # Handle empty filtered data
    if len(filtered_df) == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data matches the selected filters", showarrow=False)
        empty_table = html.Div("No data matches the selected filters", style={'textAlign': 'center', 'padding': '20px'})
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_table
    
    # Chart 1: Edge Case Distribution (Pie)
    edge_case_counts = filtered_df['edge_case_type'].value_counts()
    fig1 = px.pie(
        values=edge_case_counts.values,
        names=edge_case_counts.index,
        title='Distribution of Edge Case Types',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    # Chart 2: Severity Distribution (Histogram)
    fig2 = px.histogram(
        filtered_df,
        x='severity',
        nbins=30,
        title='Severity Distribution',
        labels={'severity': 'Severity (m/s²)'},
        color_discrete_sequence=['#636EFA']
    )
    fig2.update_layout(showlegend=False)
    
    # Chart 3: Severity by Type (Box Plot)
    fig3 = px.box(
        filtered_df,
        x='edge_case_type',
        y='severity',
        title='Severity Range by Edge Case Type',
        labels={'severity': 'Severity (m/s²)', 'edge_case_type': 'Edge Case Type'},
        color='edge_case_type'
    )
    fig3.update_layout(showlegend=False)
    
    # Chart 4: Edge Cases by File (Bar)
    file_counts = filtered_df['file_name'].value_counts().head(10)
    fig4 = px.bar(
        x=file_counts.values,
        y=file_counts.index,
        orientation='h',
        title='Top 10 Files with Most Edge Cases',
        labels={'x': 'Count', 'y': 'File Name'},
        color_discrete_sequence=['#EF553B']
    )
    fig4.update_layout(showlegend=False)
    
    # Data Table
    table_df = filtered_df[[
        'frame_id', 'file_name', 'edge_case_type', 'severity', 
        'speed_min', 'speed_max', 'accel_x_min', 'accel_y_max'
    ]].copy()
    
    table_df['severity'] = table_df['severity'].round(4)
    table_df['speed_min'] = table_df['speed_min'].round(2)
    table_df['speed_max'] = table_df['speed_max'].round(2)
    table_df['accel_x_min'] = table_df['accel_x_min'].round(2)
    table_df['accel_y_max'] = table_df['accel_y_max'].round(2)
    
    # Sort by severity descending
    table_df = table_df.sort_values('severity', ascending=False)
    
    table_html = html.Table([
        html.Thead(
            html.Tr([
                html.Th(col, style={
                    'padding': '10px',
                    'textAlign': 'left',
                    'borderBottom': '2px solid #ddd',
                    'backgroundColor': '#f0f0f0',
                    'fontWeight': 'bold'
                })
                for col in table_df.columns
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(
                    str(table_df.iloc[i][col]),
                    style={'padding': '10px', 'borderBottom': '1px solid #ddd'}
                )
                for col in table_df.columns
            ], style={'backgroundColor': '#ffffff' if i % 2 == 0 else '#f9f9f9'})
            for i in range(min(100, len(table_df)))
        ])
    ], style={
        'width': '100%',
        'borderCollapse': 'collapse',
        'marginTop': '20px',
        'fontSize': '12px'
    })
    
    return fig1, fig2, fig3, fig4, table_html

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Waymo Edge Case Dashboard...")
    print("="*60)
    print(f"📊 Dashboard available at: http://localhost:8050")
    print(f"📁 Results directory: {RESULTS_DIR}")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8050)