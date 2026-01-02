import streamlit as st
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Makanan Sehat",
    page_icon="🥗",
    layout="wide"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
model = joblib.load("model_makanan.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
.title {
    font-size: 38px;
    font-weight: 700;
    color: #FFFFFF;
}
.subtitle {
    font-size: 15px;
    color: #BBBBBB;
    margin-bottom: 20px;
}
.card {
    background-color: #1E1E1E;
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 20px;
}
.result-healthy {
    background-color: #1B5E20;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    text-align: center;
}
.result-unhealthy {
    background-color: #B71C1C;
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="title">🥗 Sistem Klasifikasi Makanan Sehat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Prediksi makanan sehat dan tidak sehat berdasarkan kandungan nutrisi</div>',
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# SIDEBAR – INFORMASI MAKANAN
# ======================================================
st.sidebar.header("📋 Informasi Makanan")

nama_makanan = st.sidebar.text_input("Nama Makanan")
jenis_makanan = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Dinner", "Dessert"]
)
jenis_diet = st.sidebar.selectbox(
    "Jenis Diet",
    ["Umum", "Vegetarian", "Vegan", "Rendah Lemak", "Rendah Gula", "Tinggi Protein"]
)
rating = st.sidebar.select_slider(
    "Rating Makanan", options=[1, 2, 3, 4, 5], value=3
)

# ======================================================
# MAIN CONTENT
# ======================================================
col1, col2 = st.columns([2, 1])

# =========================
# INPUT NUTRISI
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔢 Kandungan Nutrisi")

    c1, c2, c3 = st.columns(3)

    with c1:
        kalori = round(st.number_input("Kalori", min_value=0.0, step=0.1, format="%g"), 2)
        protein = round(st.number_input("Protein (g)", min_value=0.0, step=0.1, format="%g"), 2)
        serat = round(st.number_input("Serat (g)", min_value=0.0, step=0.1, format="%g"), 2)

    with c2:
        karbohidrat = round(st.number_input("Karbohidrat (g)", min_value=0.0, step=0.1, format="%g"), 2)
        lemak = round(st.number_input("Lemak (g)", min_value=0.0, step=0.1, format="%g"), 2)
        gula = round(st.number_input("Gula (g)", min_value=0.0, step=0.1, format="%g"), 2)

    with c3:
        natrium = round(st.number_input("Natrium (mg)", min_value=0.0, step=1.0, format="%g"), 2)
        kolesterol = round(st.number_input("Kolesterol (mg)", min_value=0.0, step=1.0, format="%g"), 2)
        porsi = round(st.number_input("Porsi Sajian (g)", min_value=0.0, step=1.0, format="%g"), 2)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDIKSI
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Hasil Prediksi")

    if st.button("Prediksi Kesehatan", use_container_width=True):

        data = np.array([[ 
            kalori, protein, karbohidrat,
            lemak, serat, gula,
            natrium, kolesterol, porsi
        ]])

        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)[0]

        if pred == 1:
            st.markdown('<div class="result-healthy">✅ MAKANAN SEHAT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-unhealthy">❌ MAKANAN TIDAK SEHAT</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("📄 Ringkasan")
        st.write(f"**Nama:** {nama_makanan if nama_makanan else '-'}")
        st.write(f"**Jenis:** {jenis_makanan}")
        st.write(f"**Diet:** {jenis_diet}")
        st.write(f"**Rating:** ⭐ {rating}/5")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption("Model Random Forest tanpa resampling | Digunakan untuk keperluan akademik")
