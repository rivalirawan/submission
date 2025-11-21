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
             
        df['year'] = df.index.year
        df['month'] = df.index.month
        
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
    if 'Spring' in seasonal_trend.columns:
        seasonal_trend = seasonal_trend[['Spring', 'Summer', 'Autumn', 'Winter']]
    return seasonal_trend


# --- 4. STREAMLIT UI & LAYOUT ---

st.title("Analisis Kualitas Udara Beijing (2013-2017)")
st.caption("Eksplorasi Pola Spasial, Musiman, dan Profil Polusi")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["1. Tren Spasial & Musiman (PB1)", "2. Profil Clustering (PB2)", "3. Korelasi Kausal (PB3)"])

with tab1:
    st.header("1. Tren Spasial dan Musiman PM2.5")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Peringkat PM2.5 Rata-Rata per Stasiun")
        spatial_pollution = df_master.groupby('station')['PM2.5'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=spatial_pollution.index, y=spatial_pollution.values, palette='viridis', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)

    with col2:
        st.subheader("Rerata PM2.5 per Musim dan Tahun")
        st.dataframe(get_seasonal_trend(df_master).style.highlight_max(axis=1))
        st.subheader("Distribusi PM2.5 Berdasarkan Musim")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='season', y='PM2.5', data=df_master, order=['Spring', 'Summer', 'Autumn', 'Winter'], palette='coolwarm', ax=ax)
        st.pyplot(fig)


with tab2:
    st.header("2. Segmentasi Kualitas Udara (Rule-Based)")
    profile_summary = df_master.groupby('Air_Quality_Profile')[['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'WSPM', 'TEMP', 'PRES']].mean()
    scaler = MinMaxScaler()
    profile_normalized = pd.DataFrame(scaler.fit_transform(profile_summary), columns=profile_summary.columns, index=profile_summary.index)

    st.subheader("Heatmap Karakteristik Profil (0=Rendah, 1=Tinggi)")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(profile_normalized, annot=True, cmap='YlOrRd', fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribusi Musiman Setiap Profil")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='Air_Quality_Profile', hue='season', data=df_master, palette='Set2', order=sorted(df_master['Air_Quality_Profile'].unique()), ax=ax)
    st.pyplot(fig)


with tab3:
    st.header("3. Bukti Kausal: Prekursor vs. Sekunder")
    corr_vars = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    corr_matrix = df_master[corr_vars].corr()
    
    st.subheader("Heatmap Korelasi Antar Polutan")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    sample_df = df_master.sample(n=2000, random_state=42)
    st.subheader("Hubungan NO2 vs PM2.5")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x='NO2', y='PM2.5', data=sample_df, scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax)
    st.pyplot(fig)
