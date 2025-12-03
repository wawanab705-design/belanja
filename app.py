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
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🏥 Analisis Prediksi Biaya Pelayanan Pasien 2025</h1>
    <p>Analisis data transaksi pelayanan pasien Jan-Nov 2025 dengan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=100)
    st.title("⚙️ Pengaturan Model")
    
    st.subheader("Parameter Model")
    test_size = st.slider("Persentase Data Uji", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    st.subheader("Model yang Akan Digunakan")
    use_lr = st.checkbox("Linear Regression", value=True)
    use_rf = st.checkbox("Random Forest", value=True)
    
    rf_n_estimators = st.slider("Jumlah Tree (Random Forest)", 50, 500, 100, 50) if use_rf else 100
    
    st.subheader("Tampilan Data")
    show_raw_data = st.checkbox("Tampilkan Data Mentah", value=False)
    n_rows_preview = st.slider("Jumlah Baris Preview", 5, 100, 10)

# Fungsi untuk load data dari GitHub
@st.cache_data
def load_data_from_github():
    """
    Load data dari URL GitHub
    """
    github_raw_url = 'https://raw.githubusercontent.com/wawanab705-design/belanja/refs/heads/main/belanja-jan-nov2025.csv'
    
    try:
        # Untuk file CSV dengan pemisah ,
        df = pd.read_csv(github_raw_url, sep=',', header=None, low_memory=False, encoding='utf-8')

        if df.shape[1] == 12:
            df.columns = [
                'id_transaksi', 'id_pasien', 'no_urut', 'nama_pasien', 'waktu',
                'dokter', 'jenis_layanan', 'poli', 'sumber_pembayaran', 'biaya',
                'diskon', 'flag'
            ]
            return df, "✅ Data berhasil dimuat dari GitHub!"
        else:
            return None, f"❌ Jumlah kolom tidak sesuai. Ditemukan {df.shape[1]} kolom."

    except Exception as e:
        return None, f"❌ Error membaca file: {str(e)}"

# Fungsi preprocessing
def preprocess_data(df):
    """
    Preprocessing data untuk modeling
    """
    # Buat salinan dataframe
    df_clean = df.copy()
    
    # Drop baris pertama (header deskriptif)
    df_clean = df_clean.iloc[1:].copy()
    
    # Konversi waktu
    df_clean['waktu'] = pd.to_datetime(df_clean['waktu'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Konversi biaya ke numeric
    df_clean['biaya'] = df_clean['biaya'].astype(str).str.replace(',', '', regex=False)
    df_clean['biaya'] = pd.to_numeric(df_clean['biaya'], errors='coerce')
    
    # Hapus baris dengan waktu atau biaya NaN
    df_clean = df_clean.dropna(subset=['waktu', 'biaya']).copy()
    
    # Ekstrak fitur waktu
    df_clean['bulan'] = df_clean['waktu'].dt.month
    df_clean['hari_dlm_minggu'] = df_clean['waktu'].dt.dayofweek
    df_clean['hari_dlm_bulan'] = df_clean['waktu'].dt.day
    
    # Encode fitur kategorikal
    label_encoders = {}
    kategori_cols = ['dokter', 'poli', 'jenis_layanan']
    
    for col in kategori_cols:
        le = LabelEncoder()
        df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    
    return df_clean, label_encoders

# Fungsi untuk membuat visualisasi
def create_visualizations(df, y_test, y_pred_lr, y_pred_rf):
    """
    Membuat visualisasi untuk dashboard
    """
    visualizations = {}
    
    # 1. Distribusi Biaya
    fig1 = px.histogram(df, x='biaya', nbins=50, 
                       title='Distribusi Biaya Pelayanan',
                       labels={'biaya': 'Biaya (Rp)', 'count': 'Jumlah Pasien'})
    fig1.update_layout(template='plotly_white')
    visualizations['distribusi_biaya'] = fig1
    
    # 2. Biaya per Bulan
    biaya_per_bulan = df.groupby('bulan')['biaya'].mean().reset_index()
    fig2 = px.line(biaya_per_bulan, x='bulan', y='biaya',
                   title='Rata-rata Biaya per Bulan',
                   markers=True)
    fig2.update_layout(xaxis_title='Bulan', yaxis_title='Rata-rata Biaya (Rp)')
    visualizations['biaya_per_bulan'] = fig2
    
    # 3. Scatter plot prediksi vs aktual (jika ada)
    if y_pred_lr is not None and len(y_test) > 0:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=y_test.values[:1000], y=y_pred_lr[:1000],
                                 mode='markers', name='Linear Regression',
                                 marker=dict(color='blue', opacity=0.6)))
        if y_pred_rf is not None:
            fig3.add_trace(go.Scatter(x=y_test.values[:1000], y=y_pred_rf[:1000],
                                     mode='markers', name='Random Forest',
                                     marker=dict(color='red', opacity=0.6)))
        fig3.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', name='Ideal',
                                 line=dict(color='green', dash='dash')))
        fig3.update_layout(title='Prediksi vs Aktual',
                          xaxis_title='Biaya Aktual (Rp)',
                          yaxis_title='Biaya Prediksi (Rp)',
                          template='plotly_white')
        visualizations['prediksi_vs_aktual'] = fig3
    
    # 4. Top 10 Poli berdasarkan jumlah pasien
    top_poli = df['poli'].value_counts().head(10).reset_index()
    top_poli.columns = ['Poli', 'Jumlah Pasien']
    fig4 = px.bar(top_poli, x='Jumlah Pasien', y='Poli', orientation='h',
                  title='Top 10 Poli Berdasarkan Jumlah Pasien',
                  color='Jumlah Pasien')
    fig4.update_layout(template='plotly_white')
    visualizations['top_poli'] = fig4
    
    return visualizations

# Main app
def main():
    # Tab navigasi
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview Data", "🤖 Modeling", "📈 Visualisasi", "🎯 Prediksi"])
    
    with tab1:
        st.header("📊 Overview Data")
        
        # Load data dengan progress bar
        with st.spinner("Memuat data dari GitHub..."):
            df, message = load_data_from_github()
        
        if df is not None:
            st.success(message)
            
            # Tampilkan metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transaksi", f"{len(df):,}")
            with col2:
                st.metric("Jumlah Kolom", len(df.columns))
            with col3:
                biaya_total = pd.to_numeric(df['biaya'].astype(str).str.replace(',', '', regex=False), errors='coerce').sum()
                st.metric("Total Biaya", f"Rp {biaya_total:,.0f}")
            with col4:
                st.metric("Periode Data", "Jan-Nov 2025")
            
            # Tampilkan preview data
            if show_raw_data:
                st.subheader("Preview Data")
                st.dataframe(df.head(n_rows_preview), use_container_width=True)
            
            # Tampilkan info dataframe
            st.subheader("Informasi Dataset")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Struktur Data")
                st.write(f"Shape: {df.shape}")
                st.write(f"Memori: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            with col2:
                st.markdown("#### Tipe Data")
                type_counts = df.dtypes.value_counts()
                for dtype, count in type_counts.items():
                    st.write(f"{dtype}: {count} kolom")
            
            # Tampilkan statistik deskriptif
            st.subheader("Statistik Deskriptif")
            try:
                # Coba konversi kolom numerik untuk statistik
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.info("Tidak ada kolom numerik untuk statistik deskriptif")
            except:
                st.warning("Tidak dapat menampilkan statistik deskriptif")
        
        else:
            st.error(message)
            return
    
    # Preprocessing untuk tab lainnya
    if df is not None:
        with st.spinner("Memproses data..."):
            df_processed, label_encoders = preprocess_data(df)
        
        # Persiapan data untuk modeling
        feature_cols = ['bulan', 'hari_dlm_minggu', 'hari_dlm_bulan',
                       'dokter_encoded', 'poli_encoded', 'jenis_layanan_encoded']
        
        if all(col in df_processed.columns for col in feature_cols):
            X = df_processed[feature_cols]
            y = df_processed['biaya']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            with tab2:
                st.header("🤖 Modeling")
                
                col1, col2 = st.columns(2)
                
                # Linear Regression
                if use_lr:
                    with col1:
                        st.markdown("### Linear Regression")
                        with st.spinner("Training Linear Regression..."):
                            lr = LinearRegression()
                            lr.fit(X_train, y_train)
                            y_pred_lr = lr.predict(X_test)
                            
                            # Metrics
                            mae_lr = mean_absolute_error(y_test, y_pred_lr)
                            rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
                            r2_lr = r2_score(y_test, y_pred_lr)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🏆 Performa Linear Regression</h4>
                                <p>MAE: Rp {mae_lr:,.2f}</p>
                                <p>RMSE: Rp {rmse_lr:,.2f}</p>
                                <p>R² Score: {r2_lr:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Random Forest
                if use_rf:
                    with col2:
                        st.markdown("### Random Forest")
                        with st.spinner(f"Training Random Forest dengan {rf_n_estimators} trees..."):
                            rf = RandomForestRegressor(n_estimators=rf_n_estimators, random_state=random_state)
                            rf.fit(X_train, y_train)
                            y_pred_rf = rf.predict(X_test)
                            
                            # Metrics
                            mae_rf = mean_absolute_error(y_test, y_pred_rf)
                            rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
                            r2_rf = r2_score(y_test, y_pred_rf)
                            
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🌲 Performa Random Forest</h4>
                                <p>MAE: Rp {mae_rf:,.2f}</p>
                                <p>RMSE: Rp {rmse_rf:,.2f}</p>
                                <p>R² Score: {r2_rf:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Perbandingan model
                if use_lr and use_rf:
                    st.subheader("📊 Perbandingan Model")
                    
                    comparison_data = {
                        'Model': ['Linear Regression', 'Random Forest'],
                        'MAE': [mae_lr, mae_rf],
                        'RMSE': [rmse_lr, rmse_rf],
                        'R²': [r2_lr, r2_rf]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Tentukan model terbaik
                    best_model = 'Random Forest' if r2_rf > r2_lr else 'Linear Regression'
                    st.success(f"✅ Model terbaik: **{best_model}** dengan R² Score: {max(r2_lr, r2_rf):.4f}")
            
            with tab3:
                st.header("📈 Visualisasi")
                
                # Buat visualisasi
                with st.spinner("Membuat visualisasi..."):
                    visualizations = create_visualizations(df_processed, y_test, 
                                                          y_pred_lr if use_lr else None, 
                                                          y_pred_rf if use_rf else None)
                
                # Tampilkan visualisasi
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'distribusi_biaya' in visualizations:
                        st.plotly_chart(visualizations['distribusi_biaya'], use_container_width=True)
                    
                    if 'biaya_per_bulan' in visualizations:
                        st.plotly_chart(visualizations['biaya_per_bulan'], use_container_width=True)
                
                with col2:
                    if 'top_poli' in visualizations:
                        st.plotly_chart(visualizations['top_poli'], use_container_width=True)
                    
                    if 'prediksi_vs_aktual' in visualizations:
                        st.plotly_chart(visualizations['prediksi_vs_aktual'], use_container_width=True)
                
                # Informasi tambahan
                st.subheader("📋 Informasi Dataset Setelah Processing")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Data Training", f"{len(X_train):,} baris")
                with col2:
                    st.metric("Data Testing", f"{len(X_test):,} baris")
                with col3:
                    st.metric("Fitur", f"{len(feature_cols)} variabel")
            
            with tab4:
                st.header("🎯 Prediksi Biaya")
                
                st.markdown("""
                <div class="prediction-card">
                    <h4>🔮 Prediksi Biaya Berdasarkan Input</h4>
                    <p>Masukkan parameter untuk memprediksi biaya pelayanan:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Form input untuk prediksi
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    bulan = st.selectbox("Bulan", range(1, 13), index=10)
                    hari_dlm_minggu = st.selectbox("Hari dalam Minggu", 
                                                  range(7), 
                                                  format_func=lambda x: ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'][x])
                
                with col2:
                    hari_dlm_bulan = st.slider("Tanggal", 1, 31, 15)
                    
                    # Dropdown untuk dokter
                    if 'dokter' in label_encoders:
                        dokter_options = list(label_encoders['dokter'].classes_)
                        dokter_selected = st.selectbox("Dokter", dokter_options)
                        dokter_encoded = label_encoders['dokter'].transform([dokter_selected])[0]
                
                with col3:
                    # Dropdown untuk poli
                    if 'poli' in label_encoders:
                        poli_options = list(label_encoders['poli'].classes_)
                        poli_selected = st.selectbox("Poli", poli_options)
                        poli_encoded = label_encoders['poli'].transform([poli_selected])[0]
                    
                    # Dropdown untuk jenis layanan
                    if 'jenis_layanan' in label_encoders:
                        layanan_options = list(label_encoders['jenis_layanan'].classes_)
                        layanan_selected = st.selectbox("Jenis Layanan", layanan_options)
                        layanan_encoded = label_encoders['jenis_layanan'].transform([layanan_selected])[0]
                
                # Tombol prediksi
                if st.button("🚀 Prediksi Biaya", type="primary"):
                    # Siapkan input
                    input_data = np.array([[bulan, hari_dlm_minggu, hari_dlm_bulan,
                                          dokter_encoded, poli_encoded, layanan_encoded]])
                    
                    # Lakukan prediksi
                    if use_lr:
                        pred_lr = lr.predict(input_data)[0]
                    
                    if use_rf:
                        pred_rf = rf.predict(input_data)[0]
                    
                    # Tampilkan hasil
                    st.subheader("📊 Hasil Prediksi")
                    
                    col1, col2 = st.columns(2)
                    
                    if use_lr:
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5>Linear Regression</h5>
                                <h3>Rp {pred_lr:,.0f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if use_rf:
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5>Random Forest</h5>
                                <h3>Rp {pred_rf:,.0f}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Rekomendasi
                    if use_lr and use_rf:
                        pred_avg = (pred_lr + pred_rf) / 2
                        st.info(f"💰 **Rata-rata Prediksi Biaya: Rp {pred_avg:,.0f}**")
                        
                        # Tambahkan insight
                        st.subheader("💡 Insight")
                        st.write(f"Berdasarkan input yang diberikan, prediksi biaya pelayanan untuk:")
                        st.write(f"- **Bulan**: {bulan}")
                        st.write(f"- **Tanggal**: {hari_dlm_bulan}")
                        st.write(f"- **Dokter**: {dokter_selected}")
                        st.write(f"- **Poli**: {poli_selected}")
                        st.write(f"- **Jenis Layanan**: {layanan_selected}")
        
        else:
            st.error("Gagal memproses data untuk modeling")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Analisis Prediksi Biaya Pelayanan Pasien 2025 | Dibuat dengan Streamlit</p>
    <p>Data Source: GitHub Repository</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
