import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Analisis Kualitas Udara Beijing",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    GOOGLE_DRIVE_FILE_ID = '1qPaH1qnOYSrRu7lsL-PQq9nCkg6pvZIa' 
    
    data_url = f'https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}'

    st.info("Memuat data dari Google Drive...")
    
    try:
        df = pd.read_csv(data_url, index_col=0, parse_dates=True)
        
        if df.empty:
             st.error("Data berhasil diunduh, tetapi DataFrame kosong.")
             return None
             
        if 'season' not in df.columns:
            df['month'] = df.index.month
            def determine_season(month):
                if month in [3, 4, 5]: return 'Spring'
                elif month in [6, 7, 8]: return 'Summer'
                elif month in [9, 10, 11]: return 'Autumn'
                else: return 'Winter'
            df['season'] = df['month'].apply(determine_season)

        if 'Air_Quality_Profile' not in df.columns:
            bins = [0, 35, 75, 115, 150, 250, np.inf]
            labels = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
            df['Air_Quality_Profile'] = pd.cut(df['PM2.5'], bins=bins, labels=labels, right=False).astype(str)
            df['Air_Quality_Profile'] = df['Air_Quality_Profile'].fillna('Hazardous')

        df['year'] = df.index.year
        
        st.success("Data berhasil dimuat!")
        return df
        
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Drive. Pastikan link publik dan ID file sudah benar. Error: {e}")
        return None

df_master = load_data()
if df_master is None:
    st.stop()
    

# --- 3. HELPER FUNCTION ---
def get_seasonal_trend(df):
    seasonal_trend = df.groupby(['year', 'season'])['PM2.5'].mean().unstack()
    season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
    existing_seasons = [s for s in season_order if s in seasonal_trend.columns]
    return seasonal_trend[existing_seasons]


# --- 4. STREAMLIT SIDEBAR (FILTERING) ---

st.sidebar.header("⚙️ Filter Analisis")

# Filter Tahun (Slider)
min_year = int(df_master['year'].min())
max_year = int(df_master['year'].max())
selected_year_range = st.sidebar.slider(
    "Pilih Rentang Tahun",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter Stasiun (Multiselect dengan "Select All")
all_stations = sorted(df_master['station'].unique())
selected_stations = st.sidebar.multiselect(
    "Pilih Stasiun",
    options=["Select All"] + all_stations,
    default=["Select All"]
)
if "Select All" in selected_stations:
    stations_to_filter = all_stations
else:
    stations_to_filter = selected_stations

# Filter Musim (Multiselect dengan "Select All")
all_seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
selected_seasons = st.sidebar.multiselect(
    "Pilih Musim",
    options=["Select All"] + all_seasons,
    default=["Select All"]
)
if "Select All" in selected_seasons:
    seasons_to_filter = all_seasons
else:
    seasons_to_filter = selected_seasons

df_filtered = df_master[
    (df_master['year'] >= selected_year_range[0]) & 
    (df_master['year'] <= selected_year_range[1]) &
    (df_master['station'].isin(stations_to_filter)) &
    (df_master['season'].isin(seasons_to_filter))
]

if df_filtered.empty:
    st.warning("DataFrame kosong! Tidak ada data yang sesuai dengan kriteria filter yang dipilih. Silakan ubah filter.")
    st.stop()


# --- 5. STREAMLIT UI & LAYOUT ---

st.title("Analisis Kualitas Udara Beijing (2013-2017)")
st.caption(f"Data ditampilkan untuk **Tahun {selected_year_range[0]}-{selected_year_range[1]}**, **{len(stations_to_filter)} Stasiun**, dan **{len(seasons_to_filter)} Musim**.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["1. Tren Spasial & Musiman (PB1)", "2. Profil Clustering (PB2)", "3. Korelasi Kausal (PB3)"])

with tab1:
    st.header("1. Tren Spasial dan Musiman PM2.5")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Peringkat PM2.5 Rata-Rata per Stasiun")
        spatial_pollution = df_filtered.groupby('station')['PM2.5'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=spatial_pollution.index, y=spatial_pollution.values, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title("Rata-Rata PM2.5 per Stasiun (Terfilter)")
        st.pyplot(fig)

    with col2:
        st.subheader("Rerata PM2.5 per Musim dan Tahun")
        st.dataframe(get_seasonal_trend(df_filtered).style.highlight_max(axis=1))
        
        st.subheader("Distribusi PM2.5 Berdasarkan Musim")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='season', y='PM2.5', data=df_filtered, order=all_seasons, palette='coolwarm', ax=ax)
        st.pyplot(fig)


with tab2:
    st.header("2. Segmentasi Kualitas Udara (Rule-Based)")
    profile_summary = df_filtered.groupby('Air_Quality_Profile')[['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'WSPM', 'TEMP', 'PRES']].mean()
    
    if profile_summary.shape[0] < 2:
        st.info("Tidak cukup data profil setelah difilter untuk membuat Heatmap yang berarti (kurang dari 2 profil).")
    else:
        show_heatmap = st.checkbox("Tampilkan Heatmap Karakteristik Profil", value=True)
        if show_heatmap:
            scaler = MinMaxScaler()
            profile_normalized = pd.DataFrame(scaler.fit_transform(profile_summary), columns=profile_summary.columns, index=profile_summary.index)
        
            st.subheader("Heatmap Karakteristik Profil (0=Rendah, 1=Tinggi)")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(profile_normalized, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title("Heatmap Karakteristik Profil Kualitas Udara (Terfilter)")
            st.pyplot(fig)

    st.subheader("Distribusi Musiman Setiap Profil")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='Air_Quality_Profile', hue='season', data=df_filtered, palette='Set2', order=sorted(df_filtered['Air_Quality_Profile'].unique()), ax=ax)
    st.pyplot(fig)


with tab3:
    st.header("3. Bukti Kausal: Prekursor vs. Sekunder")
    
    show_corr_heatmap = st.checkbox("Tampilkan Heatmap Korelasi Antar Polutan", value=True, key="corr_check")
    if show_corr_heatmap:
        corr_vars = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        corr_matrix = df_filtered[corr_vars].corr()
        
        st.subheader("Heatmap Korelasi Antar Polutan")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax)
        ax.set_title("Korelasi Polutan (Terfilter)")
        st.pyplot(fig)

    st.subheader("Hubungan NO2 vs PM2.5")
    n_samples = min(2000, len(df_filtered))
    sample_df = df_filtered.sample(n=n_samples, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='NO2', y='PM2.5', data=sample_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
    ax.set_title(f"Hubungan NO2 vs PM2.5 (Sampel {n_samples} Baris)")
    st.pyplot(fig)