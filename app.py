import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np
import os
import plotly.graph_objects as go
import msal
import requests
from dotenv import load_dotenv
import json
import datetime
import time

load_dotenv()

st.set_page_config(page_title="AQ Retail Analytics", layout="wide", page_icon="📊")

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        padding-bottom: 0.5rem;
    }

    /* Card-style metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #374151;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8fafc;
        border-radius: 8px;
        font-weight: 500;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Success/warning/error boxes */
    .stSuccess, .stWarning, .stError {
        border-radius: 8px;
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
    }

    /* Custom insight cards */
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    .insight-positive {
        border-left-color: #10b981;
    }

    .insight-negative {
        border-left-color: #ef4444;
    }

    .insight-neutral {
        border-left-color: #f59e0b;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 AQ Retail Analytics")

# Load historical sales data from CSV file
try:
    sales_df = pd.read_csv('vendite.csv')
except FileNotFoundError:
    st.sidebar.error("Missing file 'vendite.csv'. Place it in the app folder.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error reading 'vendite.csv': {e}")
    st.stop()

sales_df['data'] = pd.to_datetime(sales_df['data'], dayfirst=True, errors='coerce')
if sales_df['data'].isna().all():
    st.sidebar.error("Column 'data' could not be parsed as dates in 'vendite.csv'.")
    st.stop()

# Store list
negozi_lista = [
    "BHR", "SER", "BIR", "SRV", "LBM", "PAR", "ROM", "MIL", "ECI", "NBS", "STT", "CAP", "FLO", "NYC", "MAR", "GLA", "LRM", "BMP", "CAS"
]

# --- GLOBAL FILTERS ALWAYS DEFINED, WITH UNIQUE KEY ---
today = datetime.date.today()
def_start = today
def_end = today + datetime.timedelta(days=30)
st.sidebar.write('Select date range and stores to analyze:')
date_range = st.sidebar.date_input('Date range', value=(def_start, def_end), min_value=today - datetime.timedelta(days=365), max_value=today + datetime.timedelta(days=365), key='global_date_input')
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtro_start_date, filtro_end_date = date_range
else:
    filtro_start_date = def_start
    filtro_end_date = def_end

# Store selection combo box
selected_stores = st.sidebar.multiselect(
    'Select stores to analyze:',
    options=negozi_lista,
    default=negozi_lista,
    help='You can select one or more stores.'
)

# Function to analyze scheduling vs sales
def analyze_scheduling(sales_df, schedule_df):
    # Add day of week to sales data
    sales_df['giorno_settimana'] = sales_df['data'].dt.day_name()
    
    # Calculate average sales by day of week
    avg_sales_by_day = sales_df.groupby('giorno_settimana')['vendite'].mean()
    
    # Add day of week to schedule data
    schedule_df['giorno_settimana'] = pd.to_datetime(schedule_df['data']).dt.day_name()
    
    # Calculate staff count by day
    staff_by_day = schedule_df.groupby('giorno_settimana')['num_persone'].sum()
    
    # Calculate sales per staff member
    sales_per_staff = avg_sales_by_day / staff_by_day
    
    # Identify anomalies (days with significantly different sales/staff ratio)
    mean_sales_per_staff = sales_per_staff.mean()
    std_sales_per_staff = sales_per_staff.std()
    
    anomalies = []
    for day in sales_per_staff.index:
        ratio = sales_per_staff[day]
        if abs(ratio - mean_sales_per_staff) > 1.5 * std_sales_per_staff:
            anomalies.append({
                'giorno': day,
                'vendite_medie': avg_sales_by_day[day],
                'staff_programmato': staff_by_day[day],
                'vendite_per_persona': ratio,
                'deviazione': (ratio - mean_sales_per_staff) / std_sales_per_staff
            })
    
    return pd.DataFrame(anomalies)

# Function to analyze scheduling vs sales per DATA
def analyze_scheduling_by_date(sales_df, schedule_df):
    # Merge vendite e persone per data
    vendite_per_data = sales_df.groupby('data')['vendite'].sum().reset_index()
    persone_per_data = schedule_df.groupby('data')['num_persone'].sum().reset_index()
    merged = pd.merge(vendite_per_data, persone_per_data, on='data', how='inner')
    merged['vendite_per_persona'] = merged['vendite'] / merged['num_persone']
    # Calcolo media e std
    mean_ratio = merged['vendite_per_persona'].mean()
    std_ratio = merged['vendite_per_persona'].std()
    # Anomalia: rapporto molto diverso dalla media
    anomalies = merged[abs(merged['vendite_per_persona'] - mean_ratio) > 1.5 * std_ratio].copy()
    anomalies['deviazione_std'] = (anomalies['vendite_per_persona'] - mean_ratio) / std_ratio
    return anomalies[['data', 'vendite', 'num_persone', 'vendite_per_persona', 'deviazione_std']]

def week_of_month(dt):
    first_day = dt.replace(day=1)
    dom = dt.day
    adjusted_dom = dom + first_day.weekday()
    return int((adjusted_dom - 1) / 7) + 1

# Funzione per preparare i dati per confronto per settimana e giorno della settimana
def prepare_weekday_week_data(df, date_col):
    df = df.copy()
    df['mese'] = df[date_col].dt.month
    df['anno'] = df[date_col].dt.year
    df['giorno_settimana'] = df[date_col].dt.day_name()
    df['settimana_mese'] = df[date_col].apply(week_of_month)
    df['num_persone'] = 1  # Ogni riga = una persona
    return df

# Funzione di analisi allineata per settimana e giorno della settimana
def analyze_by_weekday_week(sales_df, schedule_df):
    vendite = prepare_weekday_week_data(sales_df, 'data')
    turni = prepare_weekday_week_data(schedule_df, 'data')
    # Media vendite storiche per combinazione
    vendite_grouped = vendite.groupby(['mese', 'giorno_settimana', 'settimana_mese'])['vendite'].mean().reset_index()
    # Media persone programmate per combinazione
    turni_grouped = turni.groupby(['mese', 'giorno_settimana', 'settimana_mese'])['num_persone'].sum().reset_index()
    # Merge
    merged = pd.merge(vendite_grouped, turni_grouped, on=['mese', 'giorno_settimana', 'settimana_mese'], how='outer').fillna(0)
    merged['vendite_per_persona'] = merged['vendite'] / merged['num_persone'].replace(0, np.nan)
    # Anomalia: rapporto molto diverso dalla media
    mean_ratio = merged['vendite_per_persona'].mean()
    std_ratio = merged['vendite_per_persona'].std()
    merged['deviazione_std'] = (merged['vendite_per_persona'] - mean_ratio) / std_ratio
    anomalies = merged[abs(merged['vendite_per_persona'] - mean_ratio) > 1.5 * std_ratio]
    return merged, anomalies

# Analisi: confronto turni con media vendite storiche per combinazione (mese, settimana, giorno)
def analyze_turni_vs_vendite(sales_df, schedule_df):
    # Prepara turni: una riga per data, con persone programmate
    turni = schedule_df.copy()
    turni['mese'] = turni['data'].dt.month
    turni['settimana_mese'] = turni['data'].apply(week_of_month)
    turni['giorno_settimana'] = turni['data'].dt.day_name()
    # Prepara vendite storiche: media per mese, settimana, giorno
    vendite = sales_df.copy()
    vendite['mese'] = vendite['data'].dt.month
    vendite['settimana_mese'] = vendite['data'].apply(week_of_month)
    vendite['giorno_settimana'] = vendite['data'].dt.day_name()
    vendite_grouped = vendite.groupby(['mese', 'settimana_mese', 'giorno_settimana'])['vendite'].mean().reset_index()
    # Merge: per ogni data dei turni, associa la media vendite storiche corrispondente
    result = pd.merge(turni, vendite_grouped, on=['mese', 'settimana_mese', 'giorno_settimana'], how='left')
    result = result[['data', 'num_persone', 'vendite', 'mese', 'settimana_mese', 'giorno_settimana']]
    result['vendite_per_persona'] = result['vendite'] / result['num_persone']
    return result

def get_weather_description(wmo_code):
    if wmo_code in [0, 1]: return "☀️"  # Clear, Mainly clear
    if wmo_code in [2]: return "🌤️"  # Partly cloudy
    if wmo_code in [3]: return "☁️"   # Overcast
    if wmo_code in [45, 48]: return "🌫️" # Fog
    if wmo_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]: return "🌧️" # Rain
    if wmo_code in [71, 73, 75, 77, 85, 86]: return "❄️"  # Snow
    if wmo_code in [95, 96, 99]: return "⛈️" # Thunderstorm
    return ""

# Dopo aver caricato sales_df
sales_df['negozio'] = sales_df['negozio'].astype(str).str.upper()
sales_df['data'] = pd.to_datetime(sales_df['data'], errors='coerce').dt.strftime('%Y-%m-%d')

# Instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. View **Sales Analytics** below for insights
2. Set date range for coverage analysis
3. Select stores to analyze
4. Click **Launch** to fetch Teams shift data
""")

# ============================================
# SALES ANALYTICS DASHBOARD
# ============================================

# Create a copy for analytics
analytics_df = sales_df.copy()
analytics_df['data'] = pd.to_datetime(analytics_df['data'], errors='coerce')
analytics_df = analytics_df.dropna(subset=['data'])
analytics_df['month'] = analytics_df['data'].dt.to_period('M')
analytics_df['month_name'] = analytics_df['data'].dt.strftime('%b %Y')
analytics_df['weekday'] = analytics_df['data'].dt.day_name()
analytics_df['week'] = analytics_df['data'].dt.isocalendar().week
analytics_df['year'] = analytics_df['data'].dt.year

# Filter for selected stores
if selected_stores:
    analytics_filtered = analytics_df[analytics_df['negozio'].isin(selected_stores)]
else:
    analytics_filtered = analytics_df

# Main tabs
tab1, tab2, tab3 = st.tabs(["📈 Sales Analytics", "🏪 Store Performance", "📅 Coverage Analysis"])

with tab1:
    st.header("Sales Overview")

    # Top-level KPIs
    total_sales = analytics_filtered['vendite'].sum()
    avg_daily_sales = analytics_filtered.groupby('data')['vendite'].sum().mean()
    num_stores = analytics_filtered['negozio'].nunique()
    date_range_str = f"{analytics_filtered['data'].min().strftime('%d %b %Y')} - {analytics_filtered['data'].max().strftime('%d %b %Y')}"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Sales", f"€{total_sales:,.0f}")
    with kpi2:
        st.metric("Avg Daily Sales", f"€{avg_daily_sales:,.0f}")
    with kpi3:
        st.metric("Active Stores", f"{num_stores}")
    with kpi4:
        st.metric("Data Period", date_range_str)

    st.markdown("---")

    # Monthly Trend
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Sales Trend")
        monthly_sales = analytics_filtered.groupby('month_name')['vendite'].sum().reset_index()
        # Sort by actual date
        monthly_sales['sort_date'] = pd.to_datetime(monthly_sales['month_name'], format='%b %Y')
        monthly_sales = monthly_sales.sort_values('sort_date')

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Scatter(
            x=monthly_sales['month_name'],
            y=monthly_sales['vendite'],
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#667eea'),
            hovertemplate='%{x}<br>Sales: €%{y:,.0f}<extra></extra>'
        ))
        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Sales (€)",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col2:
        st.subheader("Sales by Day of Week")
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = analytics_filtered.groupby('weekday')['vendite'].mean().reindex(weekday_order).reset_index()
        weekday_sales.columns = ['weekday', 'avg_sales']

        # Color gradient based on sales
        max_sales = weekday_sales['avg_sales'].max()
        colors = [f'rgba(102, 126, 234, {0.4 + 0.6 * (s / max_sales)})' for s in weekday_sales['avg_sales']]

        fig_weekday = go.Figure(go.Bar(
            x=weekday_sales['weekday'].str[:3],
            y=weekday_sales['avg_sales'],
            marker_color=colors,
            hovertemplate='%{x}<br>Avg: €%{y:,.0f}<extra></extra>'
        ))
        fig_weekday.update_layout(
            xaxis_title="",
            yaxis_title="Avg Sales (€)",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig_weekday, use_container_width=True)

    # Insights row
    st.subheader("Key Insights")
    ins_col1, ins_col2, ins_col3 = st.columns(3)

    # Best day of week
    best_day = weekday_sales.loc[weekday_sales['avg_sales'].idxmax(), 'weekday']
    worst_day = weekday_sales.loc[weekday_sales['avg_sales'].idxmin(), 'weekday']

    # Month over month growth
    if len(monthly_sales) >= 2:
        latest_month = monthly_sales.iloc[-1]['vendite']
        prev_month = monthly_sales.iloc[-2]['vendite']
        mom_growth = ((latest_month - prev_month) / prev_month) * 100 if prev_month > 0 else 0
    else:
        mom_growth = 0

    with ins_col1:
        growth_class = "insight-positive" if mom_growth > 0 else "insight-negative" if mom_growth < 0 else "insight-neutral"
        growth_icon = "📈" if mom_growth > 0 else "📉" if mom_growth < 0 else "➡️"
        st.markdown(f"""
        <div class="insight-card {growth_class}">
            <h4>{growth_icon} Month-over-Month</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{mom_growth:+.1f}%</p>
            <p style="color: #6b7280; margin: 0;">vs previous month</p>
        </div>
        """, unsafe_allow_html=True)

    with ins_col2:
        st.markdown(f"""
        <div class="insight-card insight-positive">
            <h4>🔥 Best Performing Day</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{best_day}</p>
            <p style="color: #6b7280; margin: 0;">highest average sales</p>
        </div>
        """, unsafe_allow_html=True)

    with ins_col3:
        st.markdown(f"""
        <div class="insight-card insight-neutral">
            <h4>💡 Opportunity Day</h4>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">{worst_day}</p>
            <p style="color: #6b7280; margin: 0;">lowest average - potential for growth</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.header("Store Performance")

    # Store rankings
    store_performance = analytics_filtered.groupby('negozio').agg(
        total_sales=('vendite', 'sum'),
        avg_daily_sales=('vendite', 'mean'),
        days_active=('data', 'nunique'),
        best_day=('vendite', 'max')
    ).reset_index().sort_values('total_sales', ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Store Rankings")
        # Top performers bar chart
        fig_stores = go.Figure()
        colors_gradient = [f'rgba(102, 126, 234, {1 - i * 0.05})' for i in range(len(store_performance))]
        fig_stores.add_trace(go.Bar(
            y=store_performance['negozio'],
            x=store_performance['total_sales'],
            orientation='h',
            marker_color=colors_gradient,
            hovertemplate='%{y}<br>Total: €%{x:,.0f}<extra></extra>'
        ))
        fig_stores.update_layout(
            yaxis=dict(autorange='reversed'),
            xaxis_title="Total Sales (€)",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            height=400
        )
        st.plotly_chart(fig_stores, use_container_width=True)

    with col2:
        st.subheader("Sales Distribution Heatmap")
        # Heatmap: stores x weekday
        heatmap_data = analytics_filtered.groupby(['negozio', 'weekday'])['vendite'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='negozio', columns='weekday', values='vendite')
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(columns=weekday_order)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[d[:3] for d in weekday_order],
            y=heatmap_pivot.index,
            colorscale='Purples',
            hovertemplate='%{y} - %{x}<br>Avg Sales: €%{z:,.0f}<extra></extra>'
        ))
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Store comparison table
    st.subheader("Store Metrics Comparison")
    store_display = store_performance.copy()
    store_display.columns = ['Store', 'Total Sales (€)', 'Avg Daily Sales (€)', 'Days Active', 'Best Single Day (€)']
    store_display['Total Sales (€)'] = store_display['Total Sales (€)'].apply(lambda x: f"€{x:,.0f}")
    store_display['Avg Daily Sales (€)'] = store_display['Avg Daily Sales (€)'].apply(lambda x: f"€{x:,.0f}")
    store_display['Best Single Day (€)'] = store_display['Best Single Day (€)'].apply(lambda x: f"€{x:,.0f}")
    st.dataframe(store_display, use_container_width=True, hide_index=True)

    # Top/Bottom performers
    top3 = store_performance.head(3)['negozio'].tolist()
    bottom3 = store_performance.tail(3)['negozio'].tolist()

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.markdown(f"""
        <div class="insight-card insight-positive">
            <h4>🏆 Top Performers</h4>
            <p style="font-size: 1.2rem; margin: 0;">1. {top3[0] if len(top3) > 0 else 'N/A'}</p>
            <p style="font-size: 1rem; color: #6b7280; margin: 0;">2. {top3[1] if len(top3) > 1 else 'N/A'} | 3. {top3[2] if len(top3) > 2 else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
    with perf_col2:
        st.markdown(f"""
        <div class="insight-card insight-negative">
            <h4>📊 Needs Attention</h4>
            <p style="font-size: 1.2rem; margin: 0;">{bottom3[-1] if len(bottom3) > 0 else 'N/A'}</p>
            <p style="font-size: 1rem; color: #6b7280; margin: 0;">Lower sales - review staffing & marketing</p>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Teams Integration")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")
tenant_id = os.getenv("AZURE_TENANT_ID")

# Se le credenziali non sono presenti, chiedile all'utente
if not (client_id and client_secret and tenant_id):
    st.sidebar.warning("Azure credentials (Client ID, Client Secret, Tenant ID) not found in .env. Please provide them below.")
    client_id = st.sidebar.text_input("Azure Client ID", value=client_id or "", type="default")
    client_secret = st.sidebar.text_input("Azure Client Secret", value=client_secret or "", type="password")
    tenant_id = st.sidebar.text_input("Azure Tenant ID", value=tenant_id or "", type="default")

scarica_teams = st.sidebar.button("Launch")

if scarica_teams:
    if not (client_id and client_secret and tenant_id):
        st.sidebar.error("Azure credentials (Client ID, Client Secret, Tenant ID) are required.")
    else:
        with st.spinner("Downloading data from Teams Shifts for selected stores..."):
            authority = f"https://login.microsoftonline.com/{tenant_id}"
            scope = ["https://graph.microsoft.com/.default"]
            app_auth = msal.ConfidentialClientApplication(
                client_id,
                authority=authority,
                client_credential=client_secret,
            )
            result = app_auth.acquire_token_for_client(scopes=scope)
            if "access_token" in result:
                access_token = result["access_token"]
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Content-Type': 'application/json'
                }
                # Load store -> team_id mapping
                try:
                    with open('team_mapping.json', 'r') as f:
                        team_mapping = json.load(f)
                except Exception as e:
                    st.error(f"Error loading team_mapping.json: {e}")
                    team_mapping = {}

                negozio_timezones = {}
                all_shifts = []
                progress_bar = st.progress(0, text="Downloading shifts...")

                # Use only selected stores for download
                stores_to_download = selected_stores if selected_stores else negozi_lista
                for idx, negozio in enumerate(stores_to_download):
                    team_id = team_mapping.get(negozio, "")
                    if not team_id:
                        continue

                    # Fetch timezone for the team
                    url_schedule = f"https://graph.microsoft.com/v1.0/teams/{team_id}/schedule"
                    try:
                        response_schedule = requests.get(url_schedule, headers=headers)
                        if response_schedule.status_code == 200:
                            schedule_data = response_schedule.json()
                            negozio_timezones[negozio] = schedule_data.get('timeZone')
                        else:
                            st.warning(f"Unable to retrieve timezone for {negozio}. UTC will be used.")
                            negozio_timezones[negozio] = "UTC"
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Connection error while retrieving timezone for {negozio}: {e}. UTC will be used.")
                        negozio_timezones[negozio] = "UTC"

                    url_shifts = f"https://graph.microsoft.com/v1.0/teams/{team_id}/schedule/shifts"
                    while url_shifts:
                        try:
                            response_shifts = requests.get(url_shifts, headers=headers)
                        except requests.exceptions.RequestException as e:
                            st.error(f"Errore di connessione per {negozio}: {e}")
                            break
                        if response_shifts.status_code == 200:
                            data = response_shifts.json()
                            shifts = data.get('value', [])
                            for shift in shifts:
                                start, end = None, None
                                shared_shift = shift.get('sharedShift')
                                draft_shift = shift.get('draftShift')
                                
                                # Prioritize shared (published) shift, fallback to draft
                                if shared_shift and shared_shift.get('startDateTime'):
                                    start = shared_shift.get('startDateTime')
                                    end = shared_shift.get('endDateTime')
                                elif draft_shift and draft_shift.get('startDateTime'):
                                    start = draft_shift.get('startDateTime')
                                    end = draft_shift.get('endDateTime')

                                user_id = shift.get('userId')
                                all_shifts.append({
                                    'negozio': negozio,
                                    'team_id': team_id,
                                    'user_id': user_id,
                                    'start': start,
                                    'end': end
                                })
                            url_shifts = data.get('@odata.nextLink')
                            if url_shifts:
                                time.sleep(0.5)  # Pause between requests
                        else:
                            st.warning(f"Error retrieving shifts for {negozio}: {response_shifts.text}")
                            break
                    progress_bar.progress((idx + 1) / len(negozi_lista), text=f"{negozio} completed")
                progress_bar.empty()
                if all_shifts:
                    df_shifts = pd.DataFrame(all_shifts)
                    df_shifts = df_shifts.dropna(subset=['start', 'end'])
                    df_shifts['start'] = pd.to_datetime(df_shifts['start'], utc=True)
                    df_shifts['end'] = pd.to_datetime(df_shifts['end'], utc=True)
                    # Convert to local time for each store
                    converted_dfs = []
                    for store, tz in negozio_timezones.items():
                        if tz:
                            store_df = df_shifts[df_shifts['negozio'] == store].copy()
                            if not store_df.empty:
                                store_df['start'] = store_df['start'].dt.tz_convert(tz)
                                store_df['end'] = store_df['end'].dt.tz_convert(tz)
                                converted_dfs.append(store_df)
                    stores_with_tz = negozio_timezones.keys()
                    other_stores_df = df_shifts[~df_shifts['negozio'].isin(stores_with_tz)]
                    if not other_stores_df.empty:
                        converted_dfs.append(other_stores_df)
                    if converted_dfs:
                        df_shifts = pd.concat(converted_dfs, ignore_index=True)
                    # Filter by dates and non-empty user_id
                    df_shifts['start_date_local'] = df_shifts['start'].apply(lambda x: x.date())
                    df_shifts = df_shifts[(df_shifts['start_date_local'] >= filtro_start_date) & (df_shifts['start_date_local'] <= filtro_end_date) & (df_shifts['user_id'].astype(str).str.strip() != '')]
                    if df_shifts.empty:
                        st.warning("No data available for the selected interval and/or non-empty user_id.")
                    else:
                        # Aggregation by store and date (date only, string format YYYY-MM-DD)
                        df_shifts['data'] = df_shifts['start'].apply(lambda x: x.strftime('%Y-%m-%d'))
                        df_shifts['negozio'] = df_shifts['negozio'].astype(str).str.upper()
                        schedule_df = df_shifts.groupby(['negozio', 'data'])['user_id'].count().reset_index(name='num_persone')
                        schedule_df['negozio'] = schedule_df['negozio'].astype(str).str.upper()
                        schedule_df['data'] = schedule_df['data'].astype(str)
                        st.session_state['df_shifts'] = df_shifts.copy()
                        st.session_state['schedule_df'] = schedule_df.copy()
                        st.header('Scheduled staff vs Sales per store')
                        st.success("Shift data downloaded successfully!")
                        with st.expander("Preview of downloaded data", expanded=False):
                            st.dataframe(df_shifts.head(20))

# --- Coverage Analysis in Tab 3 ---
with tab3:
    st.header("Coverage Analysis")

    has_schedule_data = 'schedule_df' in st.session_state and not st.session_state['schedule_df'].empty

    if not has_schedule_data:
        st.info("Click **Launch** in the sidebar to fetch Teams shift data and analyze staffing coverage vs expected sales.")

if has_schedule_data:
    with tab3:
        schedule_df = st.session_state['schedule_df']

        # Filter by selected stores
        if selected_stores:
            schedule_df = schedule_df[schedule_df['negozio'].isin(selected_stores)]
            sales_df_filtered = sales_df[sales_df['negozio'].isin(selected_stores)]
        else:
            sales_df_filtered = sales_df

        coverage_df = None
        if 'df_shifts' in st.session_state:
            df_shifts = st.session_state['df_shifts']
            df_shifts = df_shifts[df_shifts['negozio'].isin(selected_stores)] if selected_stores else df_shifts
            df_shifts['data_date'] = df_shifts['start'].apply(lambda x: x.date())
            coverage_df = df_shifts.groupby(['negozio', 'data_date']).agg(
                prima_entrata=('start', 'min'),
                ultima_uscita=('end', 'max')
            ).reset_index()

        st.subheader('Scheduled Staff vs Sales per Store')
        for negozio in schedule_df['negozio'].unique():
            vendite_negozio = sales_df_filtered[sales_df_filtered['negozio'] == negozio].copy()
            sched_negozio = schedule_df[schedule_df['negozio'] == negozio].copy()
            if vendite_negozio.empty:
                st.warning(f"No sales data for store {negozio}. Only planning and staff graph will be shown.")
                # Show only planning/coverage and a staff graph for this store
                sched_negozio['data_dt'] = pd.to_datetime(sched_negozio['data'])
                sched_negozio['mese'] = sched_negozio['data_dt'].dt.month
                sched_negozio['anno'] = sched_negozio['data_dt'].dt.year
                sched_negozio['giorno_settimana'] = sched_negozio['data_dt'].dt.day_name()
                sched_negozio['settimana_mese'] = sched_negozio['data_dt'].apply(lambda x: (x.day - 1) // 7 + 1)
                # Plot only staff planning (num_persone) over time
                sched_negozio = sched_negozio.sort_values('data_dt')
                sched_negozio['data_label'] = sched_negozio['data_dt'].dt.strftime('%Y-%m-%d') + ' (' + sched_negozio['giorno_settimana'] + ')'
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sched_negozio['data_label'], y=sched_negozio['num_persone'], name='Scheduled staff', mode='lines+markers', marker_color='orange', hovertemplate='%{x}<br>Staff: %{y}<extra></extra>'))
                fig.update_layout(
                    title=f"{negozio} — Scheduled staff (no sales data)",
                    xaxis_title="Date (day of week)",
                    yaxis_title="Scheduled staff",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Details (advanced)"):
                    st.dataframe(sched_negozio[['data_label', 'num_persone']].rename(columns={'data_label': 'data', 'num_persone': 'scheduled_staff'}))
                if coverage_df is not None:
                    st.subheader(f"{negozio} — Opening/closing coverage")
                    negozio_coverage = coverage_df[coverage_df['negozio'] == negozio].copy()
                    if not negozio_coverage.empty:
                        negozio_coverage['prima_entrata'] = pd.to_datetime(negozio_coverage['prima_entrata']).dt.strftime('%H:%M')
                        negozio_coverage['ultima_uscita'] = pd.to_datetime(negozio_coverage['ultima_uscita']).dt.strftime('%H:%M')
                        negozio_coverage.rename(columns={'data_date': 'data'}, inplace=True)
                        with st.expander("Daily opening/closing times"):
                            st.dataframe(negozio_coverage[['data', 'prima_entrata', 'ultima_uscita']].rename(columns={'prima_entrata': 'first_in', 'ultima_uscita': 'last_out'}))
                    else:
                        st.write("No hourly coverage data available for this store.")
                continue
            # Calcola mese, settimana del mese, giorno della settimana per turnazioni (2025)
            sched_negozio['data_dt'] = pd.to_datetime(sched_negozio['data'])
            sched_negozio['mese'] = sched_negozio['data_dt'].dt.month
            sched_negozio['anno'] = sched_negozio['data_dt'].dt.year
            sched_negozio['giorno_settimana'] = sched_negozio['data_dt'].dt.day_name()
            sched_negozio['settimana_mese'] = sched_negozio['data_dt'].apply(lambda x: (x.day - 1) // 7 + 1)
            # Calcola mese, settimana del mese, giorno della settimana per vendite (2025)
            vendite_negozio['data_dt'] = pd.to_datetime(vendite_negozio['data'])
            vendite_negozio['mese'] = vendite_negozio['data_dt'].dt.month
            vendite_negozio['anno'] = vendite_negozio['data_dt'].dt.year
            vendite_negozio['giorno_settimana'] = vendite_negozio['data_dt'].dt.day_name()
            vendite_negozio['settimana_mese'] = vendite_negozio['data_dt'].apply(lambda x: (x.day - 1) // 7 + 1)
            # Media vendite storiche per combinazione (2025)
            vendite_hist = vendite_negozio[vendite_negozio['anno'] == 2025].copy()
        
            # Arricchimento con dati meteo
            try:
                with open('store_locations.json', 'r') as f:
                    locations = json.load(f)
                lat = locations[negozio]['latitude']
                lon = locations[negozio]['longitude']
                if lat != 0 and lon != 0 and not vendite_hist.empty:
                    dates_to_fetch = vendite_hist['data_dt'].unique()
                    start_date_api = pd.to_datetime(min(dates_to_fetch)).strftime('%Y-%m-%d')
                    end_date_api = pd.to_datetime(max(dates_to_fetch)).strftime('%Y-%m-%d')
                    weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date_api}&end_date={end_date_api}&hourly=weather_code&timezone=auto"
                    try:
                        weather_response = requests.get(weather_url)
                        if weather_response.status_code == 200:
                            weather_data = weather_response.json().get('hourly', {})
                            if weather_data and 'time' in weather_data and 'weather_code' in weather_data:
                                # Process hourly data to get a daily summary
                                hourly_df = pd.DataFrame({
                                    'timestamp': pd.to_datetime(weather_data['time']),
                                    'weather_code': weather_data['weather_code']
                                })
                                hourly_df['date'] = hourly_df['timestamp'].dt.date
                                daily_weather_summary = []
                                rain_codes = [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]
                                for date, group in hourly_df.groupby('date'):
                                    rain_hours = group['weather_code'].isin(rain_codes).sum()
                                    if rain_hours >= 4:
                                        daily_weather_icon = '🌧️'
                                    else:
                                        # If not mostly rainy, use the mode of the day
                                        most_common_code = group['weather_code'].mode()[0]
                                        daily_weather_icon = get_weather_description(most_common_code)
                                    daily_weather_summary.append({'data_dt': pd.to_datetime(date), 'meteo': daily_weather_icon})
                                df_weather = pd.DataFrame(daily_weather_summary)
                                # Merge with sales data
                                vendite_hist['data_dt_normalized'] = vendite_hist['data_dt'].dt.normalize()
                                vendite_hist = pd.merge(vendite_hist, df_weather[['data_dt', 'meteo']], left_on='data_dt_normalized', right_on='data_dt', how='left')
                                vendite_hist.drop(columns=['data_dt_normalized', 'data_dt_y'], inplace=True, errors='ignore')
                                vendite_hist.rename(columns={'data_dt_x': 'data_dt'}, inplace=True)
                        else:
                            st.warning(f"Could not retrieve weather for {negozio}. (API Status: {weather_response.status_code})")
                    except requests.exceptions.RequestException as e:
                        st.warning(f"Connection error while retrieving weather for {negozio}: {e}")
            except (FileNotFoundError, KeyError):
                st.warning(f"No coordinates found for {negozio} in 'store_locations.json' or file not found.")
        
            if 'meteo' in vendite_hist.columns:
                vendite_grouped = vendite_hist.groupby(['mese', 'settimana_mese', 'giorno_settimana', 'negozio']).agg(
                    vendite=('vendite', 'mean'),
                    meteo=('meteo', lambda x: x.mode()[0] if not x.mode().empty and pd.notna(x.mode()[0]) else '')
                ).reset_index()
            else:
                vendite_grouped = vendite_hist.groupby(['mese', 'settimana_mese', 'giorno_settimana', 'negozio'])['vendite'].mean().reset_index()

            # Merge: per ogni data delle turnazioni, associa la media vendite storiche corrispondente
            sched_current = sched_negozio[sched_negozio['anno'].isin([2025, 2026])].copy()
            result = pd.merge(
                sched_current,
                vendite_grouped,
                on=['mese', 'settimana_mese', 'giorno_settimana', 'negozio'],
                how='left',
                suffixes=('', '_storico')
            )

            # Nuova logica per punti di attenzione
            if not result.empty and not result['vendite'].isnull().all():
                soglia_vendite_alta = result['vendite'].quantile(0.8)
                soglia_vendite_bassa = result['vendite'].quantile(0.2)
                soglia_personale_alta = result['num_persone'].quantile(0.8)
                soglia_personale_basso = result['num_persone'].quantile(0.2)

                mancanza_personale = (result['vendite'] >= soglia_vendite_alta) & (result['num_persone'] <= soglia_personale_basso)
                eccesso_personale = (result['vendite'] <= soglia_vendite_bassa) & (result['num_persone'] >= soglia_personale_alta)
            
                attenzione_idx = result[mancanza_personale | eccesso_personale].index.tolist()
            else:
                attenzione_idx = []

            result = result.sort_values('data_dt')
            result['vendite_per_persona'] = result['vendite'] / result['num_persone']

            # KPIs row
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg expected sales", f"{result['vendite'].mean():.0f}")
            with col2:
                st.metric("Avg scheduled staff", f"{result['num_persone'].mean():.1f}")
            with col3:
                st.metric("Sales per staff", f"{result['vendite_per_persona'].mean():.0f}")
            with col4:
                st.metric("Attention points", f"{len(attenzione_idx)}")

            st.subheader(f"{negozio} — Staffing vs expected sales")
            result['data_label'] = result['data_dt'].dt.strftime('%Y-%m-%d') + ' (' + result['giorno_settimana'] + ')'
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result['data_label'], y=result['vendite'], name='Expected sales (hist.)', marker_color='#667eea',
                hovertemplate='%{x}<br>Expected sales: €%{y:,.0f}<extra></extra>'
            ))

            if 'meteo' in result.columns and not result['meteo'].isnull().all():
                 fig.add_trace(go.Scatter(
                     x=result['data_label'],
                     y=result['vendite'],
                     mode='text',
                     text=result['meteo'],
                     textposition='top center',
                     textfont=dict(size=18),
                     showlegend=False
                ))
            
            fig.add_trace(go.Scatter(
                x=result['data_label'], y=result['num_persone'], name='Scheduled staff', mode='lines+markers', marker_color='#f59e0b', yaxis='y2',
                hovertemplate='%{x}<br>Staff: %{y:.0f}<extra></extra>'
            ))
            if attenzione_idx:
                fig.add_trace(go.Scatter(
                    x=result.loc[attenzione_idx, 'data_label'],
                    y=result.loc[attenzione_idx, 'num_persone'],
                    mode='markers',
                    marker=dict(color='red', size=12, symbol='star'),
                    name='Attention point',
                    yaxis='y2',
                    showlegend=True
                ))
            fig.update_layout(
                title=f"{negozio}: Staffing vs expected sales",
                xaxis_title="",
                yaxis_title="Expected sales (€)",
                yaxis2=dict(title="Scheduled staff", overlaying='y', side='right', tickformat=',d', dtick=1),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Details (advanced)"):
                st.dataframe(result)

            if coverage_df is not None:
                st.subheader(f"{negozio} — Opening/closing coverage")
                negozio_coverage = coverage_df[coverage_df['negozio'] == negozio].copy()
                if not negozio_coverage.empty:
                    negozio_coverage['prima_entrata'] = pd.to_datetime(negozio_coverage['prima_entrata']).dt.strftime('%H:%M')
                    negozio_coverage['ultima_uscita'] = pd.to_datetime(negozio_coverage['ultima_uscita']).dt.strftime('%H:%M')
                    negozio_coverage.rename(columns={'data_date': 'data'}, inplace=True)
                    with st.expander("Daily opening/closing times"):
                        st.dataframe(negozio_coverage[['data', 'prima_entrata', 'ultima_uscita']].rename(columns={'prima_entrata': 'first_in', 'ultima_uscita': 'last_out'}))
                else:
                    st.write("No hourly coverage data available for this store.")
