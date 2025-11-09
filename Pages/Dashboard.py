import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import json
import difflib
from dotenv import load_dotenv
import os

from datetime import datetime
load_dotenv()

# Get Gemini API key from environment
api_key = os.getenv("GEMINI_API_KEY")
# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title=" VizAI by Ethicallogix",
    page_icon="📊",
    layout="wide"
)

# ---------------------- STYLING ----------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }

        body { 
            background: linear-gradient(135deg, #e8f5f1 0%, #d4e9e2 100%);
        }

        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        .dashboard-title {
            background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 50%, #0d9488 100%);
            color: white;
            padding: 30px;
            border-radius: 24px;
            text-align: center;
            margin-bottom: 35px;
            box-shadow: 0 10px 40px rgba(45, 212, 191, 0.3);
            position: relative;
            overflow: hidden;
        }

        .dashboard-title::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .kpi-card {
            background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%);
            color: white;
            padding: 24px 20px;
            border-radius: 18px;
            text-align: center;
            box-shadow: 0 6px 24px rgba(45, 212, 191, 0.25);
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 32px rgba(45, 212, 191, 0.35);
        }

        .kpi-value {
            font-size: 2.8rem;
            font-weight: 700;
            margin: 8px 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .kpi-label {
            font-size: 0.95rem;
            opacity: 0.95;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        .chart-card {
            background: white;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
            margin-bottom: 24px;
            height: 100%;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(45, 212, 191, 0.1);
        }

        .chart-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        }

        .chart-title {
            color: #0f766e;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 20px;
            text-align: center;
            letter-spacing: -0.3px;
        }

        .chart-type-badge {
            display: inline-block;
            background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
            color: #065f46;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 12px;
            border: 1px solid #6ee7b7;
        }

        .stButton>button {
            background: linear-gradient(135deg, #2dd4bf 0%, #14b8a6 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 14px 32px;
            font-weight: 600;
            font-size: 1.05rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 16px rgba(45, 212, 191, 0.3);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(45, 212, 191, 0.4);
        }

        .info-section {
            background: white;
            padding: 24px;
            border-radius: 18px;
            border-left: 5px solid #14b8a6;
            margin: 20px 0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        }

        /* Plotly chart improvements */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- COLOR SCHEMES ----------------------
# Professional gradient color schemes
MALE_COLOR = '#8B4513'
FEMALE_COLOR = '#14b8a6'
YES_COLOR = '#10b981'
NO_COLOR = '#ef4444'

# Premium color palettes
COLOR_PALETTE_1 = ['#14b8a6', '#06b6d4', '#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b']
COLOR_PALETTE_2 = ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef', '#ec4899', '#f43f5e']
COLOR_PALETTE_3 = ['#f59e0b', '#f97316', '#ef4444', '#ec4899', '#d946ef', '#a855f7']
COLOR_PALETTE_4 = ['#10b981', '#14b8a6', '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1']
COLOR_PALETTE_5 = ['#8b5cf6', '#a855f7', '#c026d3', '#d946ef', '#ec4899', '#f43f5e']
COLOR_PALETTE_6 = ['#0ea5e9', '#06b6d4', '#14b8a6', '#10b981', '#84cc16', '#eab308']

CHART_PALETTES = {
    'bar': COLOR_PALETTE_1,
    'line': COLOR_PALETTE_4,
    'scatter': COLOR_PALETTE_2,
    'pie': COLOR_PALETTE_3,
    'histogram': COLOR_PALETTE_6,
    'box': COLOR_PALETTE_5,
    'area': COLOR_PALETTE_4,
    'donut': COLOR_PALETTE_3,
    'sunburst': COLOR_PALETTE_2,
    'treemap': COLOR_PALETTE_1
}

# ---------------------- SIDEBAR ----------------------
with st.sidebar:

    st.divider()

    st.markdown("### ⚙️ Chart Settings")
    chart_height = st.slider("Chart Height", 350, 650, 480)
    num_charts = st.slider("Number of Charts", 4, 8, 6)
    show_data_preview = st.checkbox("Show Data Preview", value=False)

    st.divider()
    st.markdown("### 🎨 Style Options")
    chart_theme = st.selectbox("Chart Style", ["Modern", "Minimal", "Dark"])

    st.divider()
    st.caption("⚡ Powered by Gemini 2.5 Flash")


# ---------------------- HELPER FUNCTIONS ----------------------
def detect_column_types(df):
    """Detect and categorize column types"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    for col in categorical_cols[:]:
        if df[col].nunique() < 10:  # Likely categorical
            continue
        try:
            pd.to_datetime(df[col], errors='raise')
            datetime_cols.append(col)
            categorical_cols.remove(col)
        except:
            pass

    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }


def find_best_match(column_name, df_columns):
    if not column_name:
        return None
    matches = difflib.get_close_matches(column_name.lower(), [c.lower() for c in df_columns], n=1, cutoff=0.5)
    if matches:
        for c in df_columns:
            if c.lower() == matches[0]:
                return c
    return None


def get_color_palette(chart_type, color_column=None, df=None):
    if color_column and df is not None:
        color_lower = color_column.lower()

        if 'sex' in color_lower or 'gender' in color_lower:
            unique_vals = df[color_column].unique()
            if len(unique_vals) == 2:
                color_map = {}
                for val in unique_vals:
                    val_str = str(val).lower()
                    if val_str in ['male', 'm', 'man', '1']:
                        color_map[val] = MALE_COLOR
                    elif val_str in ['female', 'f', 'woman', '0']:
                        color_map[val] = FEMALE_COLOR
                return color_map

        if df[color_column].nunique() == 2:
            unique_vals = df[color_column].unique()
            val_strs = [str(v).lower() for v in unique_vals]
            if any(x in val_strs for x in ['yes', 'no', 'true', 'false']):
                color_map = {}
                for val in unique_vals:
                    val_str = str(val).lower()
                    if val_str in ['yes', 'true', '1', 'active']:
                        color_map[val] = YES_COLOR
                    else:
                        color_map[val] = NO_COLOR
                return color_map

    return CHART_PALETTES.get(chart_type, COLOR_PALETTE_1)


def apply_chart_styling(fig, chart_type, theme="Modern"):
    """Apply professional styling to charts"""

    # Base configuration
    fig.update_layout(
        template="plotly_white",
        height=chart_height,
        margin=dict(l=50, r=50, t=60, b=50),
        paper_bgcolor="rgba(255,255,255,0.95)",
        plot_bgcolor="rgba(248,250,252,0.5)",
        font=dict(family='Inter, sans-serif', size=12, color='#1e293b'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
            font=dict(size=11)
        ),
        title_font=dict(size=16, color='#0f766e', family='Inter'),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter"
        )
    )

    # Enhanced grid styling
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148,163,184,0.15)',
        showline=True,
        linewidth=2,
        linecolor='rgba(148,163,184,0.2)',
        tickfont=dict(size=11, color='#475569')
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(148,163,184,0.15)',
        showline=True,
        linewidth=2,
        linecolor='rgba(148,163,184,0.2)',
        tickfont=dict(size=11, color='#475569')
    )

    # Chart-specific enhancements
    if chart_type in ['bar', 'histogram']:
        fig.update_traces(
            marker_line_color='rgba(255,255,255,0.8)',
            marker_line_width=1.5,
            opacity=0.9
        )
    elif chart_type == 'line':
        fig.update_traces(
            line_width=3,
            mode='lines+markers',
            marker_size=6
        )
    elif chart_type == 'scatter':
        fig.update_traces(
            marker_size=10,
            marker_line_width=1.5,
            marker_line_color='white'
        )
    elif chart_type in ['pie', 'donut']:
        fig.update_traces(
            textposition='inside',
            textfont_size=12,
            marker=dict(line=dict(color='white', width=2))
        )

    return fig


def create_smart_chart(df, chart_type, x, y, color, title):
    """Create professionally styled chart with error handling"""
    try:
        # Validate inputs
        if not x and not y:
            return None

        # Clean column names
        plot_df = df.copy()

        # Handle datetime conversion
        if x and x in plot_df.columns:
            if plot_df[x].dtype == 'object':
                try:
                    plot_df[x] = pd.to_datetime(plot_df[x], errors='coerce')
                except:
                    pass

        # Remove rows with NaN in key columns
        key_cols = [c for c in [x, y, color] if c and c in plot_df.columns]
        if key_cols:
            plot_df = plot_df.dropna(subset=key_cols)

        if len(plot_df) == 0:
            return None

        # Get colors
        colors = get_color_palette(chart_type, color, plot_df)
        fig = None

        # Create charts with proper validation
        if chart_type == "bar" and x and y and x in plot_df.columns and y in plot_df.columns:
            # Aggregate if too many categories
            if plot_df[x].nunique() > 20:
                agg_df = plot_df.groupby(x, as_index=False)[y].sum()
                agg_df = agg_df.nlargest(15, y)
                plot_df = agg_df

            if color and color in plot_df.columns and isinstance(colors, dict):
                fig = px.bar(plot_df, x=x, y=y, color=color, color_discrete_map=colors)
            elif color and color in plot_df.columns:
                fig = px.bar(plot_df, x=x, y=y, color=color, color_discrete_sequence=colors)
            else:
                fig = px.bar(plot_df, x=x, y=y, color_discrete_sequence=colors)

        elif chart_type == "line" and x and y and x in plot_df.columns and y in plot_df.columns:
            # Sort by x for proper line rendering
            plot_df = plot_df.sort_values(by=x)

            if color and color in plot_df.columns and isinstance(colors, dict):
                fig = px.line(plot_df, x=x, y=y, color=color, color_discrete_map=colors)
            elif color and color in plot_df.columns:
                fig = px.line(plot_df, x=x, y=y, color=color, color_discrete_sequence=colors)
            else:
                fig = px.line(plot_df, x=x, y=y, color_discrete_sequence=colors)

        elif chart_type == "scatter" and x and y and x in plot_df.columns and y in plot_df.columns:
            if color and color in plot_df.columns and isinstance(colors, dict):
                fig = px.scatter(plot_df, x=x, y=y, color=color, color_discrete_map=colors)
            elif color and color in plot_df.columns:
                fig = px.scatter(plot_df, x=x, y=y, color=color, color_discrete_sequence=colors)
            else:
                fig = px.scatter(plot_df, x=x, y=y, color_discrete_sequence=colors)

        elif chart_type == "pie" and x and y and x in plot_df.columns and y in plot_df.columns:
            # Aggregate and limit categories
            pie_df = plot_df.groupby(x, as_index=False)[y].sum()
            pie_df = pie_df.nlargest(8, y)

            palette = colors if not isinstance(colors, dict) else COLOR_PALETTE_3
            fig = px.pie(pie_df, names=x, values=y, color_discrete_sequence=palette)

        elif chart_type == "donut" and x and y and x in plot_df.columns and y in plot_df.columns:
            donut_df = plot_df.groupby(x, as_index=False)[y].sum()
            donut_df = donut_df.nlargest(8, y)

            palette = colors if not isinstance(colors, dict) else COLOR_PALETTE_3
            fig = px.pie(donut_df, names=x, values=y, hole=0.45, color_discrete_sequence=palette)

        elif chart_type == "histogram" and x and x in plot_df.columns:
            if color and color in plot_df.columns and isinstance(colors, dict):
                fig = px.histogram(plot_df, x=x, color=color, color_discrete_map=colors, nbins=25)
            elif color and color in plot_df.columns:
                fig = px.histogram(plot_df, x=x, color=color, color_discrete_sequence=colors, nbins=25)
            else:
                fig = px.histogram(plot_df, x=x, color_discrete_sequence=colors, nbins=25)

        elif chart_type == "box":
            if y and y in plot_df.columns:
                if color and color in plot_df.columns and isinstance(colors, dict):
                    fig = px.box(plot_df, x=x if x in plot_df.columns else None, y=y, color=color,
                                 color_discrete_map=colors)
                elif color and color in plot_df.columns:
                    fig = px.box(plot_df, x=x if x in plot_df.columns else None, y=y, color=color,
                                 color_discrete_sequence=colors)
                else:
                    fig = px.box(plot_df, x=x if x in plot_df.columns else None, y=y, color_discrete_sequence=colors)

        elif chart_type == "area" and x and y and x in plot_df.columns and y in plot_df.columns:
            plot_df = plot_df.sort_values(by=x)

            if color and color in plot_df.columns and isinstance(colors, dict):
                fig = px.area(plot_df, x=x, y=y, color=color, color_discrete_map=colors)
            elif color and color in plot_df.columns:
                fig = px.area(plot_df, x=x, y=y, color=color, color_discrete_sequence=colors)
            else:
                fig = px.area(plot_df, x=x, y=y, color_discrete_sequence=colors)

        elif chart_type == "sunburst" and x and y and x in plot_df.columns and y in plot_df.columns:
            sunburst_df = plot_df.groupby(x, as_index=False)[y].sum()
            path = [x]
            if color and color in plot_df.columns and color != x:
                sunburst_df = plot_df.groupby([x, color], as_index=False)[y].sum()
                path = [x, color]
            fig = px.sunburst(sunburst_df, path=path, values=y, color_discrete_sequence=COLOR_PALETTE_2)

        elif chart_type == "treemap" and x and y and x in plot_df.columns and y in plot_df.columns:
            treemap_df = plot_df.groupby(x, as_index=False)[y].sum()
            path = [x]
            if color and color in plot_df.columns and color != x:
                treemap_df = plot_df.groupby([x, color], as_index=False)[y].sum()
                path = [x, color]
            fig = px.treemap(treemap_df, path=path, values=y, color_discrete_sequence=COLOR_PALETTE_1)

        if fig:
            fig = apply_chart_styling(fig, chart_type, chart_theme)
            # Remove undefined/null labels
            fig.update_layout(
                xaxis_title=x if x else "",
                yaxis_title=y if y else "",
            )

        return fig
    except Exception as e:
        st.warning(f"⚠️ Could not create {chart_type} chart: {str(e)}")
        return None


def create_kpi_card(label, value, icon="📊"):
    return f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{icon} {label}</div>
            <div class='kpi-value'>{value}</div>
        </div>
    """


# ---------------------- MAIN APP ----------------------
st.markdown("""
    <div class='dashboard-title'>
        <h1 style='margin:0; font-size: 3rem; position: relative; z-index: 1;'>📊 AI CHART DASHBOARD</h1>
        <p style='margin:12px 0 0 0; opacity: 0.95; font-size: 1.1rem; position: relative; z-index: 1;'>Transform your data into beautiful insights</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload CSV or Excel file", type=["csv", "xlsx", "xls"],
                                 help="Maximum file size: 200MB")

if uploaded_file and api_key:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Successfully loaded {df.shape[0]:,} rows × {df.shape[1]} columns")

        col_types = detect_column_types(df)

        # KPI Section
        st.markdown("### 📊 Dataset Insights")
        kpi_cols = st.columns(5)

        icons = ["📈", "📋", "🔢", "📝", "⚠️"]
        labels = ["Total Rows", "Total Columns", "Numeric", "Categorical", "Missing Data"]
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        values = [f"{df.shape[0]:,}", f"{df.shape[1]}", f"{len(col_types['numeric'])}",
                  f"{len(col_types['categorical'])}", f"{missing_pct:.1f}%"]

        for col, icon, label, value in zip(kpi_cols, icons, labels, values):
            with col:
                st.markdown(create_kpi_card(label, value, icon), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if show_data_preview:
            with st.expander("📋 Data Preview", expanded=False):
                st.dataframe(df.head(15), use_container_width=True, height=400)

        # Generate Dashboard Button
        if st.button("✨ Generate AI Dashboard", use_container_width=True, type="primary"):
            with st.spinner("🎨 Creating your dashboard..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel("models/gemini-2.5-flash")

                    column_summary = []
                    for col in df.columns:
                        col_info = f"- {col}: {str(df[col].dtype)} | {df[col].nunique()} unique"
                        if df[col].dtype in ['int64', 'float64']:
                            col_info += f" | Range: {df[col].min():.1f}-{df[col].max():.1f}"
                        column_summary.append(col_info)

                    prompt = f"""
                    Create {num_charts} DIFFERENT visualizations for this dataset. Each must use a UNIQUE chart type.

                    Available types: bar, line, scatter, pie, histogram, box, area, donut, sunburst, treemap

                    Dataset:
                    {chr(10).join(column_summary)}

                    Return ONLY a JSON array with {num_charts} objects:
                    [{{"chart_type": "bar", "title": "Sales by Region", "x_axis": "region", "y_axis": "sales", "color": "category", "description": "Shows sales distribution"}}]

                    Rules:
                    - NO duplicate chart types
                    - Use exact column names
                    - Most insightful combinations
                    - Brief descriptions
                    """

                    response = model.generate_content(prompt)
                    raw = response.text.strip()

                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]

                    suggestions = json.loads(raw.strip())

                    st.markdown("---")
                    st.markdown("### 📊 Your AI-Generated Dashboard")

                    for idx in range(0, len(suggestions), 2):
                        chart_row = st.columns(2, gap="large")

                        for col_idx, chart_col in enumerate(chart_row):
                            chart_idx = idx + col_idx
                            if chart_idx < len(suggestions):
                                s = suggestions[chart_idx]

                                with chart_col:
                                    st.markdown("<div class='chart-card'>", unsafe_allow_html=True)
                                    st.markdown(
                                        f"<span class='chart-type-badge'>{s.get('chart_type', 'chart').upper()}</span>",
                                        unsafe_allow_html=True)
                                    st.markdown(f"<div class='chart-title'>{s.get('title', 'Chart')}</div>",
                                                unsafe_allow_html=True)

                                    x = find_best_match(s.get("x_axis"), df.columns)
                                    y = find_best_match(s.get("y_axis"), df.columns)
                                    color = find_best_match(s.get("color"), df.columns)

                                    fig = create_smart_chart(df, s.get("chart_type", "").lower(), x, y, color,
                                                             s.get('title', ''))

                                    if not fig:
                                        numeric_cols = col_types['numeric']
                                        if len(numeric_cols) >= 2:
                                            fig = create_smart_chart(df, 'scatter', numeric_cols[0], numeric_cols[1],
                                                                     None, "Data Overview")

                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                                    st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    except Exception as e:
        st.error(f"❌ File error: {str(e)}")

else:
    st.markdown("""
    <div class='info-section'>
        <h3 style='margin-top:0; color:#0f766e;'>🚀 Get Started</h3>
        <p style='font-size:1.05rem; line-height:1.6; color:#475569;'>
        Upload your CSV or Excel file and enter your Gemini API key to generate beautiful, insightful charts automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='text-align:center; padding:20px;'>
            <h3 style='color:#0f766e;'>🎨 Professional Design</h3>
            <p style='color:#64748b;'>Beautiful, modern charts with smart color schemes</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align:center; padding:20px;'>
            <h3 style='color:#0f766e;'>🤖 AI-Powered</h3>
            <p style='color:#64748b;'>Intelligent analysis by Gemini 2.5 Flash</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align:center; padding:20px;'>
            <h3 style='color:#0f766e;'>⚡ Lightning Fast</h3>
            <p style='color:#64748b;'>Generate dashboards in seconds</p>
        </div>
        """, unsafe_allow_html=True)