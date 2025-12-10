import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Biaya Pasien 2025",
    page_icon="🏥",
    layout="wide"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        color: #1E3A8A;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .filter-section {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #d1e7ff;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🏥 Analisis Biaya Pelayanan Pasien 2025</h1>
    <p>Analisis data transaksi pelayanan pasien Jan-Nov 2025</p>
</div>
""", unsafe_allow_html=True)

# Fungsi untuk load data dari GitHub
@st.cache_data
def load_data_from_github():
    """
    Load data dari URL GitHub
    """
    # URL untuk dataset pertama (default)
    github_raw_url_1 = 'https://raw.githubusercontent.com/wawanab705-design/belanja/refs/heads/main/belanja-jan-nov2025.csv'
    
    # URL untuk dataset baru dengan jenis jaminan
    github_raw_url_2 = 'https://raw.githubusercontent.com/wawanab705-design/belanja/refs/heads/main/belanja-pasien-asuransi2025.csv'
    
    try:
        # Load dataset pertama
        df1 = pd.read_csv(github_raw_url_1, sep=',', header=None, low_memory=False, encoding='utf-8')
        
        # Load dataset kedua dengan jenis jaminan
        df2 = pd.read_csv(github_raw_url_2, sep=';', encoding='utf-8')
        
        # Proses dataset pertama
        if df1.shape[1] == 12:
            df1.columns = [
                'id_transaksi', 'id_pasien', 'no_urut', 'nama_pasien', 'waktu',
                'dokter', 'jenis_layanan', 'poli', 'sumber_pembayaran', 'biaya',
                'diskon', 'flag'
            ]
        else:
            return None, None, f"❌ Dataset 1: Jumlah kolom tidak sesuai. Ditemukan {df1.shape[1]} kolom."
        
        # Proses dataset kedua - mapping kolom ke format yang sama
        column_mapping = {
            'NO': 'id_transaksi',
            'RM': 'id_pasien',
            'EPS': 'no_urut',
            'NAMA': 'nama_pasien',
            'ADMISI': 'waktu',
            'DOKTER': 'dokter',
            'JENIS PELAYANAN': 'poli',
            'RAWAT': 'jenis_layanan',
            'PENJAMIN': 'jenis_jaminan',  # Kolom baru untuk jenis jaminan
            'TOTAL': 'biaya',
            'DISKON': 'diskon',
            'MENINGGAL': 'flag'
        }
        
        df2 = df2.rename(columns=column_mapping)
        
        # Tambahkan kolom sumber_pembayaran yang kosong untuk dataset kedua
        df2['sumber_pembayaran'] = df2['jenis_jaminan']
        
        # Gabungkan kedua dataset
        df_combined = pd.concat([df1, df2], ignore_index=True)
        
        return df_combined, df2, "✅ Data berhasil dimuat dari GitHub (2 dataset digabungkan)!"
        
    except Exception as e:
        return None, None, f"❌ Error membaca file: {str(e)}"

# Fungsi preprocessing
def preprocess_data(df):
    """
    Preprocessing data untuk modeling dan visualisasi
    """
    # Buat salinan dataframe
    df_clean = df.copy()
    
    # Drop baris pertama (header deskriptif) jika ada
    if 'id_transaksi' in df_clean.columns:
        # Cek apakah baris pertama adalah header deskriptif
        if not df_clean.empty and str(df_clean['id_transaksi'].iloc[0]) == 'id_transaksi':
            df_clean = df_clean.iloc[1:].copy()
    
    # Konversi waktu
    df_clean['waktu'] = pd.to_datetime(df_clean['waktu'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Konversi biaya ke numeric
    df_clean['biaya'] = df_clean['biaya'].astype(str).str.replace(',', '', regex=False)
    df_clean['biaya'] = pd.to_numeric(df_clean['biaya'], errors='coerce')
    
    # Hapus baris dengan waktu atau biaya NaN
    df_clean = df_clean.dropna(subset=['waktu', 'biaya']).copy()
    
    # Ekstrak fitur waktu
    df_clean['tanggal'] = df_clean['waktu'].dt.date
    df_clean['tahun'] = df_clean['waktu'].dt.year
    df_clean['bulan'] = df_clean['waktu'].dt.month
    df_clean['hari'] = df_clean['waktu'].dt.day
    df_clean['hari_dlm_minggu'] = df_clean['waktu'].dt.dayofweek
    df_clean['hari_dlm_bulan'] = df_clean['waktu'].dt.day
    df_clean['bulan_tahun'] = df_clean['waktu'].dt.to_period('M').astype(str)
    df_clean['minggu'] = df_clean['waktu'].dt.isocalendar().week
    
    # Tambahkan kolom jenis_jaminan jika tidak ada (untuk dataset pertama)
    if 'jenis_jaminan' not in df_clean.columns:
        df_clean['jenis_jaminan'] = df_clean.get('sumber_pembayaran', 'Tidak Diketahui')
    
    # Encode fitur kategorikal untuk modeling
    label_encoders = {}
    kategori_cols = ['dokter', 'poli', 'jenis_layanan']
    
    for col in kategori_cols:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    return df_clean, label_encoders

# Fungsi untuk filter data
def filter_data(df, start_date, end_date, selected_poli, selected_dokter, selected_jaminan, filter_type="tanggal"):
    """
    Filter data berdasarkan tanggal/bulan/tahun, poli, dokter, dan jenis jaminan
    """
    df_filtered = df.copy()
    
    # Filter berdasarkan jenis filter
    if filter_type == "tanggal" and start_date and end_date:
        df_filtered = df_filtered[(df_filtered['tanggal'] >= start_date) & 
                                 (df_filtered['tanggal'] <= end_date)]
    
    elif filter_type == "bulan" and start_date and end_date:
        # Filter berdasarkan bulan dan tahun
        df_filtered['bulan_tahun_filter'] = df_filtered['waktu'].dt.to_period('M')
        start_period = pd.Period(start_date.strftime('%Y-%m'), freq='M')
        end_period = pd.Period(end_date.strftime('%Y-%m'), freq='M')
        
        mask = (df_filtered['bulan_tahun_filter'] >= start_period) & \
               (df_filtered['bulan_tahun_filter'] <= end_period)
        df_filtered = df_filtered[mask]
    
    elif filter_type == "tahun" and start_date and end_date:
        # Filter berdasarkan tahun
        start_year = start_date.year
        end_year = end_date.year
        
        df_filtered = df_filtered[(df_filtered['tahun'] >= start_year) & 
                                 (df_filtered['tahun'] <= end_year)]
    
    # Filter berdasarkan poli
    if selected_poli and selected_poli != "Semua Poli":
        df_filtered = df_filtered[df_filtered['poli'] == selected_poli]
    
    # Filter berdasarkan dokter
    if selected_dokter and selected_dokter != "Semua Dokter":
        df_filtered = df_filtered[df_filtered['dokter'] == selected_dokter]
    
    # Filter berdasarkan jenis jaminan
    if selected_jaminan and selected_jaminan != "Semua Jaminan":
        df_filtered = df_filtered[df_filtered['jenis_jaminan'] == selected_jaminan]
    
    return df_filtered

# Fungsi untuk membuat visualisasi dengan ID unik
def create_visualizations(df, tab_name="", y_test=None, y_pred=None):
    """
    Membuat visualisasi untuk dashboard dengan ID unik berdasarkan tab_name
    """
    visualizations = {}
    
    # Buat suffix unik berdasarkan tab_name
    suffix = f"_{tab_name}" if tab_name else ""
    suffix = suffix.replace(" ", "_").lower()
    
    # 1. Distribusi Biaya Pelayanan
    if len(df) > 0:
        fig1 = px.histogram(df, x='biaya', nbins=50, 
                           title='📊 Distribusi Biaya Pelayanan',
                           labels={'biaya': 'Biaya (Rp)', 'count': 'Jumlah Pasien'},
                           color_discrete_sequence=['#3b82f6'])
        fig1.update_layout(
            template='plotly_white',
            xaxis_title="Biaya (Rupiah)",
            yaxis_title="Jumlah Pasien",
            showlegend=False,
            hovermode='x unified'
        )
        fig1.update_xaxes(
            tickformat=",.0f",
            tickprefix="Rp ",
            tickfont=dict(size=10)
        )
        fig1.update_traces(
            hovertemplate="<b>Biaya:</b> Rp %{x:,.0f}<br><b>Jumlah Pasien:</b> %{y}<extra></extra>"
        )
        visualizations[f'distribusi_biaya{suffix}'] = fig1
    
    # 2. Top 10 Poli berdasarkan jumlah pasien
    if len(df) > 0:
        top_poli = df['poli'].value_counts().head(10).reset_index()
        top_poli.columns = ['Poli', 'Jumlah Pasien']
        
        fig2 = px.bar(top_poli, x='Jumlah Pasien', y='Poli', orientation='h',
                      title='🏆 Top 10 Poli Berdasarkan Jumlah Pasien',
                      color='Jumlah Pasien',
                      color_continuous_scale='viridis',
                      text='Jumlah Pasien')
        fig2.update_layout(
            template='plotly_white',
            xaxis_title="Jumlah Pasien",
            yaxis_title="",
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig2.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="<b>Poli:</b> %{y}<br><b>Jumlah Pasien:</b> %{x:,}<extra></extra>"
        )
        visualizations[f'top_poli{suffix}'] = fig2
    
    # 3. Top 10 Dokter berdasarkan jumlah pasien
    if len(df) > 0:
        top_dokter = df['dokter'].value_counts().head(10).reset_index()
        top_dokter.columns = ['Dokter', 'Jumlah Pasien']
        
        fig2b = px.bar(top_dokter, x='Jumlah Pasien', y='Dokter', orientation='h',
                       title='👨‍⚕️ Top 10 Dokter Berdasarkan Jumlah Pasien',
                       color='Jumlah Pasien',
                       color_continuous_scale='blues',
                       text='Jumlah Pasien')
        fig2b.update_layout(
            template='plotly_white',
            xaxis_title="Jumlah Pasien",
            yaxis_title="",
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig2b.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="<b>Dokter:</b> %{y}<br><b>Jumlah Pasien:</b> %{x:,}<extra></extra>"
        )
        visualizations[f'top_dokter{suffix}'] = fig2b
    
    # 4. Distribusi Jenis Jaminan (Visualisasi Baru)
    if len(df) > 0 and 'jenis_jaminan' in df.columns:
        # Ambil top 10 jenis jaminan
        top_jaminan = df['jenis_jaminan'].value_counts().head(10).reset_index()
        top_jaminan.columns = ['Jenis Jaminan', 'Jumlah Pasien']
        
        fig2c = px.bar(top_jaminan, x='Jumlah Pasien', y='Jenis Jaminan', orientation='h',
                       title='🏥 Top 10 Jenis Jaminan',
                       color='Jumlah Pasien',
                       color_continuous_scale='greens',
                       text='Jumlah Pasien')
        fig2c.update_layout(
            template='plotly_white',
            xaxis_title="Jumlah Pasien",
            yaxis_title="Jenis Jaminan",
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        fig2c.update_traces(
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate="<b>Jenis Jaminan:</b> %{y}<br><b>Jumlah Pasien:</b> %{x:,}<extra></extra>"
        )
        visualizations[f'top_jaminan{suffix}'] = fig2c
    
    # 5. Rata-rata Biaya per Poli (Top 10)
    if len(df) > 0:
        # Ambil top 10 poli berdasarkan jumlah pasien untuk visualisasi biaya
        top_poli_list = df['poli'].value_counts().head(10).index.tolist()
        df_top_poli = df[df['poli'].isin(top_poli_list)]
        
        biaya_per_poli = df_top_poli.groupby('poli').agg({
            'biaya': 'mean',
            'id_pasien': 'count'
        }).reset_index()
        biaya_per_poli.columns = ['Poli', 'Rata-rata Biaya', 'Jumlah Pasien']
        
        # Sort by rata-rata biaya
        biaya_per_poli = biaya_per_poli.sort_values('Rata-rata Biaya', ascending=False)
        
        fig3 = px.bar(biaya_per_poli, x='Rata-rata Biaya', y='Poli', orientation='h',
                      title='💰 Rata-rata Biaya per Poli (Top 10)',
                      color='Jumlah Pasien',
                      color_continuous_scale='plasma',
                      text='Rata-rata Biaya')
        fig3.update_layout(
            template='plotly_white',
            xaxis_title="Rata-rata Biaya (Rupiah)",
            yaxis_title="",
            xaxis=dict(tickformat=",.0f", tickprefix="Rp "),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig3.update_traces(
            texttemplate='Rp %{text:,.0f}',
            textposition='outside',
            hovertemplate="<b>Poli:</b> %{y}<br><b>Rata-rata Biaya:</b> Rp %{x:,.0f}<br><b>Jumlah Pasien:</b> %{customdata[0]:,}<extra></extra>",
            customdata=np.column_stack([biaya_per_poli['Jumlah Pasien'].values])
        )
        visualizations[f'rata_biaya_per_poli{suffix}'] = fig3
    
    # 6. Rata-rata Biaya per Dokter (Top 10)
    if len(df) > 0:
        # Ambil top 10 dokter berdasarkan jumlah pasien untuk visualisasi biaya
        top_dokter_list = df['dokter'].value_counts().head(10).index.tolist()
        df_top_dokter = df[df['dokter'].isin(top_dokter_list)]
        
        biaya_per_dokter = df_top_dokter.groupby('dokter').agg({
            'biaya': 'mean',
            'id_pasien': 'count'
        }).reset_index()
        biaya_per_dokter.columns = ['Dokter', 'Rata-rata Biaya', 'Jumlah Pasien']
        
        # Sort by rata-rata biaya
        biaya_per_dokter = biaya_per_dokter.sort_values('Rata-rata Biaya', ascending=False)
        
        fig3b = px.bar(biaya_per_dokter, x='Rata-rata Biaya', y='Dokter', orientation='h',
                       title='💰 Rata-rata Biaya per Dokter (Top 10)',
                       color='Jumlah Pasien',
                       color_continuous_scale='greens',
                       text='Rata-rata Biaya')
        fig3b.update_layout(
            template='plotly_white',
            xaxis_title="Rata-rata Biaya (Rupiah)",
            yaxis_title="",
            xaxis=dict(tickformat=",.0f", tickprefix="Rp "),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig3b.update_traces(
            texttemplate='Rp %{text:,.0f}',
            textposition='outside',
            hovertemplate="<b>Dokter:</b> %{y}<br><b>Rata-rata Biaya:</b> Rp %{x:,.0f}<br><b>Jumlah Pasien:</b> %{customdata[0]:,}<extra></extra>",
            customdata=np.column_stack([biaya_per_dokter['Jumlah Pasien'].values])
        )
        visualizations[f'rata_biaya_per_dokter{suffix}'] = fig3b
    
    # 7. Rata-rata Biaya per Jenis Jaminan (Visualisasi Baru)
    if len(df) > 0 and 'jenis_jaminan' in df.columns:
        # Ambil top 10 jenis jaminan berdasarkan jumlah pasien
        top_jaminan_list = df['jenis_jaminan'].value_counts().head(10).index.tolist()
        df_top_jaminan = df[df['jenis_jaminan'].isin(top_jaminan_list)]
        
        biaya_per_jaminan = df_top_jaminan.groupby('jenis_jaminan').agg({
            'biaya': ['mean', 'sum', 'count']
        }).reset_index()
        biaya_per_jaminan.columns = ['Jenis Jaminan', 'Rata-rata Biaya', 'Total Biaya', 'Jumlah Pasien']
        
        # Sort by rata-rata biaya
        biaya_per_jaminan = biaya_per_jaminan.sort_values('Rata-rata Biaya', ascending=False)
        
        fig3c = px.bar(biaya_per_jaminan, x='Rata-rata Biaya', y='Jenis Jaminan', orientation='h',
                       title='💰 Rata-rata Biaya per Jenis Jaminan (Top 10)',
                       color='Jumlah Pasien',
                       color_continuous_scale='oranges',
                       text='Rata-rata Biaya')
        fig3c.update_layout(
            template='plotly_white',
            xaxis_title="Rata-rata Biaya (Rupiah)",
            yaxis_title="Jenis Jaminan",
            xaxis=dict(tickformat=",.0f", tickprefix="Rp "),
            yaxis={'categoryorder': 'total ascending'}
        )
        fig3c.update_traces(
            texttemplate='Rp %{text:,.0f}',
            textposition='outside',
            hovertemplate="<b>Jenis Jaminan:</b> %{y}<br><b>Rata-rata Biaya:</b> Rp %{x:,.0f}<br><b>Jumlah Pasien:</b> %{customdata[0]:,}<extra></extra>",
            customdata=np.column_stack([biaya_per_jaminan['Jumlah Pasien'].values])
        )
        visualizations[f'rata_biaya_per_jaminan{suffix}'] = fig3c
    
    # 8. Biaya per Pasien (Top 20)
    if len(df) > 0:
        # Group by nama pasien untuk melihat total biaya per pasien
        biaya_per_pasien = df.groupby('nama_pasien').agg({
            'biaya': 'sum',
            'id_pasien': 'count'
        }).reset_index()
        biaya_per_pasien.columns = ['Nama Pasien', 'Total Biaya', 'Jumlah Kunjungan']
        
        # Ambil top 20 pasien dengan total biaya tertinggi
        biaya_per_pasien = biaya_per_pasien.sort_values('Total Biaya', ascending=False).head(20)
        
        fig4 = px.bar(biaya_per_pasien, x='Total Biaya', y='Nama Pasien', orientation='h',
                      title='👤 Biaya per Pasien (Top 20)',
                      color='Jumlah Kunjungan',
                      color_continuous_scale='sunset',
                      text='Total Biaya')
        fig4.update_layout(
            template='plotly_white',
            xaxis_title="Total Biaya (Rupiah)",
            yaxis_title="Nama Pasien",
            xaxis=dict(tickformat=",.0f", tickprefix="Rp "),
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        fig4.update_traces(
            texttemplate='Rp %{text:,.0f}',
            textposition='outside',
            hovertemplate="<b>Pasien:</b> %{y}<br><b>Total Biaya:</b> Rp %{x:,.0f}<br><b>Jumlah Kunjungan:</b> %{customdata[0]:,}<extra></extra>",
            customdata=np.column_stack([biaya_per_pasien['Jumlah Kunjungan'].values])
        )
        visualizations[f'biaya_per_pasien{suffix}'] = fig4
    
    # 9. Trend Biaya berdasarkan Periode
    if len(df) > 0:
        # Agregasi biaya berdasarkan bulan
        biaya_periode = df.groupby('bulan_tahun').agg({
            'biaya': ['sum', 'mean', 'count']
        }).reset_index()
        biaya_periode.columns = ['Periode', 'Total Biaya', 'Rata-rata Biaya', 'Jumlah Pasien']
        
        # Urutkan berdasarkan periode
        biaya_periode['Periode'] = pd.to_datetime(biaya_periode['Periode'])
        biaya_periode = biaya_periode.sort_values('Periode')
        biaya_periode['Periode_Str'] = biaya_periode['Periode'].dt.strftime('%b %Y')
        
        # Gabungkan data untuk customdata
        custom_data = np.column_stack([
            biaya_periode['Rata-rata Biaya'].values,
            biaya_periode['Jumlah Pasien'].values
        ])
        
        fig5 = go.Figure()
        
        # Total biaya
        fig5.add_trace(go.Bar(
            x=biaya_periode['Periode_Str'],
            y=biaya_periode['Total Biaya'],
            name='Total Biaya',
            marker_color='#3b82f6',
            text=biaya_periode['Total Biaya'].apply(lambda x: f'Rp {x/1e6:,.1f}M' if x >= 1e6 else f'Rp {x/1e3:,.0f}K'),
            textposition='outside',
            hovertemplate="<b>Periode:</b> %{x}<br><b>Total Biaya:</b> Rp %{y:,.0f}<br><b>Rata-rata:</b> Rp %{customdata[0]:,.0f}<br><b>Jumlah Pasien:</b> %{customdata[1]:,}<extra></extra>",
            customdata=custom_data
        ))
        
        # Rata-rata biaya (line)
        fig5.add_trace(go.Scatter(
            x=biaya_periode['Periode_Str'],
            y=biaya_periode['Rata-rata Biaya'],
            name='Rata-rata Biaya',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=8),
            hovertemplate="<b>Periode:</b> %{x}<br><b>Rata-rata Biaya:</b> Rp %{y:,.0f}<extra></extra>"
        ))
        
        fig5.update_layout(
            title='📈 Trend Biaya Berdasarkan Periode',
            xaxis_title='Periode',
            yaxis_title='Total Biaya (Rupiah)',
            yaxis2=dict(
                title='Rata-rata Biaya (Rupiah)',
                overlaying='y',
                side='right',
                tickformat=",.0f",
                tickprefix="Rp "
            ),
            template='plotly_white',
            hovermode='x unified',
            barmode='group',
            yaxis=dict(
                tickformat=",.0f",
                tickprefix="Rp "
            )
        )
        visualizations[f'trend_biaya{suffix}'] = fig5
    
    # 10. Pie Chart Distribusi Jenis Jaminan (Visualisasi Baru)
    if len(df) > 0 and 'jenis_jaminan' in df.columns:
        # Hitung distribusi jenis jaminan
        jaminan_dist = df['jenis_jaminan'].value_counts().head(8).reset_index()
        jaminan_dist.columns = ['Jenis Jaminan', 'Jumlah']
        
        # Hitung "Lainnya" untuk sisa jaminan
        if len(df['jenis_jaminan'].value_counts()) > 8:
            total_lainnya = df['jenis_jaminan'].value_counts().iloc[8:].sum()
            lainnya_row = pd.DataFrame({'Jenis Jaminan': ['Lainnya'], 'Jumlah': [total_lainnya]})
            jaminan_dist = pd.concat([jaminan_dist, lainnya_row], ignore_index=True)
        
        fig5b = px.pie(jaminan_dist, values='Jumlah', names='Jenis Jaminan',
                       title='📊 Distribusi Jenis Jaminan',
                       color_discrete_sequence=px.colors.qualitative.Set3,
                       hole=0.4)
        fig5b.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persentase: %{percent}<extra></extra>"
        )
        fig5b.update_layout(
            template='plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        visualizations[f'distribusi_jaminan_pie{suffix}'] = fig5b
    
    # 11. Scatter plot prediksi vs aktual (jika ada)
    if y_pred is not None and y_test is not None and len(y_test) > 0:
        fig6 = go.Figure()
        
        # Batasi jumlah data untuk performa
        max_points = min(1000, len(y_test))
        indices = np.random.choice(len(y_test), max_points, replace=False)
        y_test_sample = y_test.iloc[indices]
        y_pred_sample = y_pred[indices]
        selisih = np.abs(y_test_sample - y_pred_sample)
        
        # Gabungkan data untuk hover
        custom_data = np.column_stack([selisih])
        
        fig6.add_trace(go.Scatter(
            x=y_test_sample, 
            y=y_pred_sample,
            mode='markers', 
            name='Prediksi vs Aktual',
            marker=dict(color='blue', opacity=0.6, size=8),
            hovertemplate="<b>Aktual:</b> Rp %{x:,.0f}<br><b>Prediksi:</b> Rp %{y:,.0f}<br><b>Selisih:</b> Rp %{customdata[0]:,.0f}<extra></extra>",
            customdata=custom_data
        ))
        
        # Garis ideal (y = x)
        min_val = min(y_test_sample.min(), y_pred_sample.min())
        max_val = max(y_test_sample.max(), y_pred_sample.max())
        
        fig6.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines', 
            name='Garis Ideal',
            line=dict(color='green', dash='dash', width=2),
            hovertemplate=None
        ))
        
        fig6.update_layout(
            title='🎯 Prediksi vs Aktual',
            xaxis_title='Biaya Aktual (Rupiah)',
            yaxis_title='Biaya Prediksi (Rupiah)',
            template='plotly_white',
            xaxis=dict(tickformat=",.0f", tickprefix="Rp "),
            yaxis=dict(tickformat=",.0f", tickprefix="Rp ")
        )
        visualizations[f'prediksi_vs_aktual{suffix}'] = fig6
    
    return visualizations

# Main app
def main():
    # Load data pertama kali
    with st.spinner("Memuat data dari GitHub..."):
        df_raw, df_jaminan, message = load_data_from_github()
    
    if df_raw is None:
        st.error(message)
        return
    
    # Preprocess data
    with st.spinner("Memproses data..."):
        df_processed, label_encoders = preprocess_data(df_raw)
    
    # Sidebar - Filter
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/hospital.png", width=100)
        st.title("🔍 Filter Data")
        
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.subheader("📅 Jenis Filter Waktu")
        
        # Pilihan jenis filter
        filter_type = st.selectbox(
            "Pilih Jenis Filter",
            options=["tanggal", "bulan", "tahun"],
            index=0,
            key="filter_type_select"
        )
        
        # Tentukan range tanggal dari data
        min_date = df_processed['tanggal'].min() if len(df_processed) > 0 else pd.Timestamp('2025-01-01').date()
        max_date = df_processed['tanggal'].max() if len(df_processed) > 0 else pd.Timestamp('2025-11-30').date()
        
        # Date range picker sesuai jenis filter
        if filter_type == "tanggal":
            st.subheader("📅 Filter Tanggal")
            start_date = st.date_input(
                "Tanggal Mulai",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key="start_date"
            )
            
            end_date = st.date_input(
                "Tanggal Akhir",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key="end_date"
            )
            
        elif filter_type == "bulan":
            st.subheader("📅 Filter Bulan")
            
            # Buat daftar bulan yang tersedia
            available_months = sorted(df_processed['bulan_tahun'].unique())
            month_options = [f"{pd.Period(m).strftime('%B %Y')}" for m in available_months]
            
            selected_months = st.multiselect(
                "Pilih Bulan",
                options=month_options,
                default=month_options[:2] if len(month_options) >= 2 else month_options,
                key="selected_months"
            )
            
            if selected_months:
                # Konversi kembali ke periode
                start_date = pd.Period(selected_months[0].split()[0][:3] + " " + selected_months[0].split()[1], freq='M').to_timestamp()
                end_date = pd.Period(selected_months[-1].split()[0][:3] + " " + selected_months[-1].split()[1], freq='M').to_timestamp()
            else:
                start_date = min_date
                end_date = max_date
                
        else:  # tahun
            st.subheader("📅 Filter Tahun")
            
            # Buat daftar tahun yang tersedia
            available_years = sorted(df_processed['tahun'].unique())
            year_options = [str(int(year)) for year in available_years]
            
            selected_years = st.multiselect(
                "Pilih Tahun",
                options=year_options,
                default=year_options,
                key="selected_years"
            )
            
            if selected_years:
                start_date = pd.Timestamp(f"{selected_years[0]}-01-01")
                end_date = pd.Timestamp(f"{selected_years[-1]}-12-31")
            else:
                start_date = min_date
                end_date = max_date
        
        # Validasi tanggal
        if start_date > end_date:
            st.warning("⚠️ Tanggal mulai harus sebelum tanggal akhir")
            start_date, end_date = min_date, max_date
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.subheader("🏥 Filter Poli")
        
        # Dapatkan daftar poli unik
        poli_options = ["Semua Poli"] + sorted(df_processed['poli'].dropna().unique().tolist())
        selected_poli = st.selectbox(
            "Pilih Poli",
            options=poli_options,
            index=0,
            key="poli_select"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.subheader("👨‍⚕️ Filter Dokter")
        
        # Dapatkan daftar dokter unik
        dokter_options = ["Semua Dokter"] + sorted(df_processed['dokter'].dropna().unique().tolist())
        selected_dokter = st.selectbox(
            "Pilih Dokter",
            options=dokter_options,
            index=0,
            key="dokter_select"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.subheader("🏢 Filter Jenis Jaminan")
        
        # Dapatkan daftar jenis jaminan unik
        if 'jenis_jaminan' in df_processed.columns:
            jaminan_options = ["Semua Jaminan"] + sorted(df_processed['jenis_jaminan'].dropna().unique().tolist())
            selected_jaminan = st.selectbox(
                "Pilih Jenis Jaminan",
                options=jaminan_options,
                index=0,
                key="jaminan_select"
            )
        else:
            st.info("Data jenis jaminan tidak tersedia")
            selected_jaminan = "Semua Jaminan"
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistik filter
        st.markdown('<div class="filter-section">', unsafe_allow_html=True)
        st.subheader("📊 Statistik Filter")
        
        # Filter data untuk statistik
        df_temp_filtered = filter_data(df_processed, start_date, end_date, selected_poli, selected_dokter, selected_jaminan, filter_type)
        
        st.write(f"**Data Awal:** {len(df_processed):,} baris")
        st.write(f"**Data Filtered:** {len(df_temp_filtered):,} baris")
        
        if len(df_temp_filtered) > 0:
            avg_biaya = df_temp_filtered['biaya'].mean()
            total_biaya = df_temp_filtered['biaya'].sum()
            st.write(f"**Rata-rata Biaya:** Rp {avg_biaya:,.0f}")
            st.write(f"**Total Biaya:** Rp {total_biaya:,.0f}")
            
            # Tampilkan info filter yang aktif
            if filter_type == "tanggal":
                st.write(f"**Periode:** {start_date} s/d {end_date}")
            elif filter_type == "bulan":
                st.write(f"**Bulan:** {selected_months if 'selected_months' in locals() else 'Semua'}")
            else:
                st.write(f"**Tahun:** {selected_years if 'selected_years' in locals() else 'Semua'}")
            
            st.write(f"**Poli:** {selected_poli}")
            st.write(f"**Dokter:** {selected_dokter}")
            if 'jenis_jaminan' in df_processed.columns:
                st.write(f"**Jenis Jaminan:** {selected_jaminan}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tombol reset filter
        if st.button("🔄 Reset Filter", type="secondary", use_container_width=True):
            st.rerun()
    
    # Filter data berdasarkan input sidebar
    df_filtered = filter_data(df_processed, start_date, end_date, selected_poli, selected_dokter, selected_jaminan, filter_type)
    
    # Tab navigasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Visualisasi Biaya", "👥 Data Pasien", "🏢 Jenis Jaminan", "🤖 Prediksi"])
    
    with tab1:
        st.header("📊 Dashboard Utama")
        
        # Info filter aktif
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if filter_type == "tanggal":
                st.metric("Periode", f"{start_date} to {end_date}")
            elif filter_type == "bulan":
                st.metric("Bulan", f"{len(selected_months) if 'selected_months' in locals() else 'Semua'} bulan")
            else:
                st.metric("Tahun", f"{len(selected_years) if 'selected_years' in locals() else 'Semua'} tahun")
        with col2:
            st.metric("Poli", selected_poli)
        with col3:
            st.metric("Dokter", selected_dokter)
        with col4:
            if 'jenis_jaminan' in df_processed.columns:
                st.metric("Jenis Jaminan", selected_jaminan)
            else:
                st.metric("Jaminan", "Tidak Tersedia")
        with col5:
            st.metric("Jumlah Data", f"{len(df_filtered):,}")
        
        # Metrics utama
        if len(df_filtered) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_biaya = df_filtered['biaya'].sum()
                st.metric("Total Biaya", f"Rp {total_biaya:,.0f}")
            with col2:
                avg_biaya = df_filtered['biaya'].mean()
                st.metric("Rata-rata Biaya", f"Rp {avg_biaya:,.0f}")
            with col3:
                max_biaya = df_filtered['biaya'].max()
                st.metric("Biaya Tertinggi", f"Rp {max_biaya:,.0f}")
            with col4:
                min_biaya = df_filtered['biaya'].min()
                st.metric("Biaya Terendah", f"Rp {min_biaya:,.0f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                jumlah_pasien = df_filtered['id_pasien'].nunique()
                st.metric("Pasien Unik", f"{jumlah_pasien:,}")
            with col2:
                jumlah_transaksi = len(df_filtered)
                st.metric("Total Transaksi", f"{jumlah_transaksi:,}")
            with col3:
                jumlah_poli = df_filtered['poli'].nunique()
                st.metric("Jumlah Poli", f"{jumlah_poli}")
            with col4:
                jumlah_dokter = df_filtered['dokter'].nunique()
                st.metric("Jumlah Dokter", f"{jumlah_dokter}")
            
            # Metrics untuk jenis jaminan jika tersedia
            if 'jenis_jaminan' in df_filtered.columns:
                col1, col2 = st.columns(2)
                with col1:
                    jumlah_jaminan = df_filtered['jenis_jaminan'].nunique()
                    st.metric("Jenis Jaminan Unik", f"{jumlah_jaminan}")
                with col2:
                    # Jaminan dengan jumlah pasien terbanyak
                    if not df_filtered.empty:
                        top_jaminan = df_filtered['jenis_jaminan'].value_counts().index[0]
                        st.metric("Jaminan Terbanyak", top_jaminan[:20] + "..." if len(top_jaminan) > 20 else top_jaminan)
    
    with tab2:
        st.header("📈 Visualisasi Data Biaya")
        
        if len(df_filtered) == 0:
            st.warning("⚠️ Tidak ada data untuk divisualisasikan dengan filter saat ini")
        else:
            # Buat visualisasi dengan suffix unik untuk tab2
            with st.spinner("Membuat visualisasi..."):
                visualizations_tab2 = create_visualizations(df_filtered, "tab2")
            
            # Tampilkan visualisasi dalam 2 kolom
            col1, col2 = st.columns(2)
            
            with col1:
                if f'distribusi_biaya_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'distribusi_biaya_tab2'], use_container_width=True, key="distribusi_biaya_tab2")
                
                if f'top_poli_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'top_poli_tab2'], use_container_width=True, key="top_poli_tab2")
                
                if f'rata_biaya_per_poli_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'rata_biaya_per_poli_tab2'], use_container_width=True, key="rata_biaya_per_poli_tab2")
                
                if f'trend_biaya_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'trend_biaya_tab2'], use_container_width=True, key="trend_biaya_tab2")
            
            with col2:
                if f'top_dokter_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'top_dokter_tab2'], use_container_width=True, key="top_dokter_tab2")
                
                if f'rata_biaya_per_dokter_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'rata_biaya_per_dokter_tab2'], use_container_width=True, key="rata_biaya_per_dokter_tab2")
                
                # Tampilkan visualisasi terkait jaminan jika ada
                if f'top_jaminan_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'top_jaminan_tab2'], use_container_width=True, key="top_jaminan_tab2")
                
                if f'distribusi_jaminan_pie_tab2' in visualizations_tab2:
                    st.plotly_chart(visualizations_tab2[f'distribusi_jaminan_pie_tab2'], use_container_width=True, key="distribusi_jaminan_pie_tab2")
    
    with tab3:
        st.header("👥 Data Biaya per Pasien")
        
        if len(df_filtered) == 0:
            st.warning("⚠️ Tidak ada data untuk ditampilkan dengan filter saat ini")
        else:
            # Buat visualisasi dengan suffix unik untuk tab3
            with st.spinner("Membuat visualisasi biaya per pasien..."):
                visualizations_tab3 = create_visualizations(df_filtered, "tab3")
            
            if f'biaya_per_pasien_tab3' in visualizations_tab3:
                st.plotly_chart(visualizations_tab3[f'biaya_per_pasien_tab3'], use_container_width=True, key="biaya_per_pasien_tab3")
            
            # Tabel detail biaya per pasien
            st.subheader("📋 Detail Biaya per Pasien")
            
            # Group by pasien
            pasien_summary = df_filtered.groupby(['nama_pasien', 'id_pasien']).agg({
                'biaya': ['sum', 'mean', 'count'],
                'poli': lambda x: ', '.join(x.unique()[:3]),
                'dokter': lambda x: ', '.join(x.unique()[:2]),
                'jenis_jaminan': lambda x: x.mode()[0] if not x.empty and len(x.mode()) > 0 else 'Tidak Diketahui'
            }).reset_index()
            
            pasien_summary.columns = ['Nama Pasien', 'ID Pasien', 'Total Biaya', 'Rata-rata Biaya', 
                                     'Jumlah Kunjungan', 'Poli', 'Dokter', 'Jenis Jaminan Utama']
            
            # Sort by total biaya
            pasien_summary = pasien_summary.sort_values('Total Biaya', ascending=False)
            
            # Format nilai Rupiah
            pasien_summary['Total Biaya Formatted'] = pasien_summary['Total Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            pasien_summary['Rata-rata Biaya Formatted'] = pasien_summary['Rata-rata Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            
            # Tampilkan tabel
            display_cols = ['Nama Pasien', 'Jumlah Kunjungan', 'Total Biaya Formatted', 
                          'Rata-rata Biaya Formatted', 'Poli', 'Dokter', 'Jenis Jaminan Utama']
            
            st.dataframe(
                pasien_summary[display_cols].rename(columns={
                    'Total Biaya Formatted': 'Total Biaya',
                    'Rata-rata Biaya Formatted': 'Rata-rata Biaya'
                }),
                use_container_width=True,
                hide_index=True,
                key="dataframe_pasien_tab3"
            )
            
            # Export option
            csv = pasien_summary[['Nama Pasien', 'ID Pasien', 'Jumlah Kunjungan', 
                                 'Total Biaya', 'Rata-rata Biaya', 'Poli', 'Dokter', 'Jenis Jaminan Utama']].to_csv(index=False)
            
            st.download_button(
                label="📥 Download Data Pasien (CSV)",
                data=csv,
                file_name="data_biaya_pasien.csv",
                mime="text/csv",
                key="download_pasien_tab3"
            )
    
    with tab4:
        st.header("🏢 Analisis Jenis Jaminan")
        
        if len(df_filtered) == 0:
            st.warning("⚠️ Tidak ada data untuk ditampilkan dengan filter saat ini")
        elif 'jenis_jaminan' not in df_filtered.columns:
            st.warning("⚠️ Data jenis jaminan tidak tersedia dalam dataset")
        else:
            # Buat visualisasi khusus jaminan dengan suffix unik
            with st.spinner("Membuat analisis jenis jaminan..."):
                visualizations_tab4 = create_visualizations(df_filtered, "tab4")
            
            # Tampilkan semua visualisasi terkait jaminan
            col1, col2 = st.columns(2)
            
            with col1:
                if f'top_jaminan_tab4' in visualizations_tab4:
                    st.plotly_chart(visualizations_tab4[f'top_jaminan_tab4'], use_container_width=True, key="top_jaminan_tab4")
                
                if f'distribusi_jaminan_pie_tab4' in visualizations_tab4:
                    st.plotly_chart(visualizations_tab4[f'distribusi_jaminan_pie_tab4'], use_container_width=True, key="distribusi_jaminan_pie_tab4")
            
            with col2:
                if f'rata_biaya_per_jaminan_tab4' in visualizations_tab4:
                    st.plotly_chart(visualizations_tab4[f'rata_biaya_per_jaminan_tab4'], use_container_width=True, key="rata_biaya_per_jaminan_tab4")
            
            # Analisis detail per jenis jaminan
            st.subheader("📊 Analisis Detail per Jenis Jaminan")
            
            # Group by jenis jaminan
            jaminan_summary = df_filtered.groupby('jenis_jaminan').agg({
                'biaya': ['sum', 'mean', 'min', 'max', 'count'],
                'id_pasien': 'nunique',
                'poli': lambda x: ', '.join(x.value_counts().head(3).index.tolist()),
                'dokter': lambda x: ', '.join(x.value_counts().head(3).index.tolist())
            }).reset_index()
            
            jaminan_summary.columns = ['Jenis Jaminan', 'Total Biaya', 'Rata-rata Biaya', 
                                      'Biaya Min', 'Biaya Max', 'Jumlah Transaksi', 
                                      'Jumlah Pasien Unik', 'Top 3 Poli', 'Top 3 Dokter']
            
            # Sort by total biaya
            jaminan_summary = jaminan_summary.sort_values('Total Biaya', ascending=False)
            
            # Format nilai Rupiah
            jaminan_summary['Total Biaya Formatted'] = jaminan_summary['Total Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            jaminan_summary['Rata-rata Biaya Formatted'] = jaminan_summary['Rata-rata Biaya'].apply(lambda x: f"Rp {x:,.0f}")
            jaminan_summary['Biaya Min Formatted'] = jaminan_summary['Biaya Min'].apply(lambda x: f"Rp {x:,.0f}")
            jaminan_summary['Biaya Max Formatted'] = jaminan_summary['Biaya Max'].apply(lambda x: f"Rp {x:,.0f}")
            
            # Tampilkan tabel
            display_cols = ['Jenis Jaminan', 'Jumlah Transaksi', 'Jumlah Pasien Unik',
                          'Total Biaya Formatted', 'Rata-rata Biaya Formatted',
                          'Biaya Min Formatted', 'Biaya Max Formatted', 'Top 3 Poli', 'Top 3 Dokter']
            
            st.dataframe(
                jaminan_summary[display_cols].rename(columns={
                    'Total Biaya Formatted': 'Total Biaya',
                    'Rata-rata Biaya Formatted': 'Rata-rata Biaya',
                    'Biaya Min Formatted': 'Biaya Min',
                    'Biaya Max Formatted': 'Biaya Max'
                }),
                use_container_width=True,
                hide_index=True,
                key="dataframe_jaminan_tab4"
            )
            
            # Export option untuk data jaminan
            csv_jaminan = jaminan_summary[['Jenis Jaminan', 'Jumlah Transaksi', 'Jumlah Pasien Unik',
                                         'Total Biaya', 'Rata-rata Biaya', 'Biaya Min', 'Biaya Max',
                                         'Top 3 Poli', 'Top 3 Dokter']].to_csv(index=False)
            
            st.download_button(
                label="📥 Download Data Jaminan (CSV)",
                data=csv_jaminan,
                file_name="data_analisis_jaminan.csv",
                mime="text/csv",
                key="download_jaminan_tab4"
            )
    
    with tab5:
        st.header("🤖 Prediksi Biaya")
        
        if len(df_filtered) < 100:
            st.warning(f"⚠️ Data terlalu sedikit ({len(df_filtered)} baris) untuk modeling. Minimal 100 data diperlukan.")
        else:
            # Persiapan data untuk modeling
            feature_cols = ['bulan', 'hari_dlm_minggu', 'hari_dlm_bulan',
                          'dokter_encoded', 'poli_encoded', 'jenis_layanan_encoded']
            
            if all(col in df_filtered.columns for col in feature_cols):
                X = df_filtered[feature_cols]
                y = df_filtered['biaya']
                
                # Split data (80-20)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                st.info(f"""
                **Info Dataset untuk Prediksi:**
                - Total Data: {len(X):,}
                - Data Training: {len(X_train):,} (80%)
                - Data Testing: {len(X_test):,} (20%)
                - Fitur: {len(feature_cols)} variabel
                """)
                
                # Gunakan Random Forest untuk prediksi
                with st.spinner("Training model untuk prediksi..."):
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_train, y_train)
                    y_pred = rf.predict(X_test)
                    
                    # Metrics
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    
                    # Tampilkan metrik (TANPA parameter key)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MAE (Mean Absolute Error)", f"Rp {mae:,.0f}")
                    with col2:
                        st.metric("RMSE (Root Mean Square Error)", f"Rp {rmse:,.0f}")
                    with col3:
                        st.metric("R² Score", f"{r2:.4f}")
                
                # Visualisasi prediksi vs aktual dengan suffix unik
                st.subheader("🎯 Visualisasi Prediksi vs Aktual")
                with st.spinner("Membuat visualisasi prediksi..."):
                    pred_viz = create_visualizations(df_filtered, "tab5", y_test, y_pred)
                
                if f'prediksi_vs_aktual_tab5' in pred_viz:
                    st.plotly_chart(pred_viz[f'prediksi_vs_aktual_tab5'], use_container_width=True, key="prediksi_vs_aktual_tab5")
                
                # Tampilkan beberapa contoh prediksi
                st.subheader("📋 Contoh Prediksi vs Aktual")
                
                # Pilih beberapa contoh acak
                sample_size = min(10, len(y_test))
                sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
                
                contoh_data = {
                    'Aktual (Rp)': y_test.iloc[sample_indices].values,
                    'Prediksi (Rp)': y_pred[sample_indices],
                    'Selisih (Rp)': np.abs(y_test.iloc[sample_indices].values - y_pred[sample_indices])
                }
                
                contoh_df = pd.DataFrame(contoh_data)
                contoh_df['Aktual (Rp)'] = contoh_df['Aktual (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
                contoh_df['Prediksi (Rp)'] = contoh_df['Prediksi (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
                contoh_df['Selisih (Rp)'] = contoh_df['Selisih (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
                
                st.dataframe(contoh_df, use_container_width=True, hide_index=True, key="dataframe_prediksi_tab5")
                
                # Interpretasi R² Score
                if r2 > 0.7:
                    st.success(f"✅ **Model memiliki performa yang baik** dengan R² Score: {r2:.4f}")
                    st.info("Model mampu menjelaskan lebih dari 70% variasi dalam data biaya.")
                elif r2 > 0.5:
                    st.info(f"ℹ️ **Model memiliki performa cukup baik** dengan R² Score: {r2:.4f}")
                    st.info("Model mampu menjelaskan lebih dari 50% variasi dalam data biaya.")
                elif r2 > 0.3:
                    st.warning(f"⚠️ **Model memiliki performa sedang** dengan R² Score: {r2:.4f}")
                    st.info("Model mampu menjelaskan lebih dari 30% variasi dalam data biaya.")
                else:
                    st.error(f"❌ **Model memiliki performa rendah** dengan R² Score: {r2:.4f}")
                    st.info("Pertimbangkan untuk menambahkan lebih banyak fitur atau data untuk meningkatkan performa model.")
            else:
                st.error("Fitur untuk modeling tidak lengkap dalam data yang difilter")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Analisis Biaya Pelayanan Pasien 2025 - Dengan Data Jenis Jaminan</p>
    <p>Data Source: 2 Dataset dari GitHub | Update Terakhir: November 2025</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
