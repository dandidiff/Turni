import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="Coverage Tool", layout="wide")

st.title("Coverage Tool")

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
st.sidebar.write('Select date range for analysis:')
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
1. Set the desired date range for the analysis (default is today to 30 days ahead).
2. Click 'Download shifts from Teams for all stores'.
3. The application will download the data, perform the analysis, and display the results for all stores.
""")

st.sidebar.header("Download shifts from Teams Shifts (Microsoft Graph API)")
client_id = os.getenv("AZURE_CLIENT_ID")
client_secret = os.getenv("AZURE_CLIENT_SECRET")
tenant_id = os.getenv("AZURE_TENANT_ID")
scarica_teams = st.sidebar.button("Download shifts from Teams for all stores")

if scarica_teams:
    if not (client_id and client_secret and tenant_id):
        st.sidebar.error("Azure credentials (Client ID, Client Secret, Tenant ID) not found. Make sure they are set in the .env file.")
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
                        st.header('Automatic analysis: Scheduled staff vs Sales per store')
                        st.success("Shift data downloaded successfully!")
                        st.write("Preview of downloaded data:")
                        st.dataframe(df_shifts.head(20))

# --- Analisi e grafici sempre disponibili se i dati sono in session_state ---
if 'schedule_df' in st.session_state and not st.session_state['schedule_df'].empty:
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

    st.header('Automatic analysis: Scheduled staff vs Sales per store')
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
        # Calcola mese, settimana del mese, giorno della settimana per vendite (2024)
        vendite_negozio['data_dt'] = pd.to_datetime(vendite_negozio['data'])
        vendite_negozio['mese'] = vendite_negozio['data_dt'].dt.month
        vendite_negozio['anno'] = vendite_negozio['data_dt'].dt.year
        vendite_negozio['giorno_settimana'] = vendite_negozio['data_dt'].dt.day_name()
        vendite_negozio['settimana_mese'] = vendite_negozio['data_dt'].apply(lambda x: (x.day - 1) // 7 + 1)
        # Media vendite storiche per combinazione (2024)
        vendite_2024 = vendite_negozio[vendite_negozio['anno'] == 2024].copy()
        
        # Arricchimento con dati meteo
        try:
            with open('store_locations.json', 'r') as f:
                locations = json.load(f)
            lat = locations[negozio]['latitude']
            lon = locations[negozio]['longitude']
            if lat != 0 and lon != 0 and not vendite_2024.empty:
                dates_to_fetch = vendite_2024['data_dt'].unique()
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
                            vendite_2024['data_dt_normalized'] = vendite_2024['data_dt'].dt.normalize()
                            vendite_2024 = pd.merge(vendite_2024, df_weather[['data_dt', 'meteo']], left_on='data_dt_normalized', right_on='data_dt', how='left')
                            vendite_2024.drop(columns=['data_dt_normalized', 'data_dt_y'], inplace=True, errors='ignore')
                            vendite_2024.rename(columns={'data_dt_x': 'data_dt'}, inplace=True)
                    else:
                        st.warning(f"Could not retrieve weather for {negozio}. (API Status: {weather_response.status_code})")
                except requests.exceptions.RequestException as e:
                    st.warning(f"Connection error while retrieving weather for {negozio}: {e}")
        except (FileNotFoundError, KeyError):
            st.warning(f"No coordinates found for {negozio} in 'store_locations.json' or file not found.")
        
        if 'meteo' in vendite_2024.columns:
            vendite_grouped = vendite_2024.groupby(['mese', 'settimana_mese', 'giorno_settimana', 'negozio']).agg(
                vendite=('vendite', 'mean'),
                meteo=('meteo', lambda x: x.mode()[0] if not x.mode().empty and pd.notna(x.mode()[0]) else '')
            ).reset_index()
        else:
            vendite_grouped = vendite_2024.groupby(['mese', 'settimana_mese', 'giorno_settimana', 'negozio'])['vendite'].mean().reset_index()

        # Merge: per ogni data delle turnazioni 2025, associa la media vendite storiche corrispondente
        sched_2025 = sched_negozio[sched_negozio['anno'] == 2025].copy()
        result = pd.merge(
            sched_2025,
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
            x=result['data_label'], y=result['vendite'], name='Expected sales (hist. 2024)', marker_color='#1f77b4',
            hovertemplate='%{x}<br>Expected sales: %{y:.0f}<extra></extra>'
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
            x=result['data_label'], y=result['num_persone'], name='Scheduled staff (2025)', mode='lines+markers', marker_color='#ff7f0e', yaxis='y2',
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
            xaxis_title="Date (day of week)",
            yaxis_title="Expected sales (2024)",
            yaxis2=dict(title="Scheduled staff (2025)", overlaying='y', side='right', tickformat=',d', dtick=1),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
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
