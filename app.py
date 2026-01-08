import streamlit as st
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Makanan Sehat",
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
    font-size: 36px;
    font-weight: 700;
    color: #FFFFFF;
}
.subtitle {
    font-size: 15px;
    color: #CCCCCC;
    margin-bottom: 25px;
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
st.markdown('<div class="title">Sistem Klasifikasi Makanan Sehat</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Klasifikasi makanan sehat dan tidak sehat berdasarkan kandungan nutrisi menggunakan Machine Learning</div>',
    unsafe_allow_html=True
)

st.divider()

# ======================================================
# SIDEBAR – INFORMASI MAKANAN
# ======================================================
st.sidebar.header("Informasi Umum Makanan")

nama_makanan = st.sidebar.text_input("Nama Makanan")

asal_makanan = st.sidebar.selectbox(
    "Asal Makanan",
    ["Lokal", "Internasional"]
)

jenis_makanan = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Dessert"]
)

prep_time_min = st.sidebar.number_input(
    "Waktu Persiapan (menit)",
    min_value=0,
    step=1,
    format="%g"
)

cook_time_min = st.sidebar.number_input(
    "Waktu Memasak (menit)",
    min_value=0,
    step=1,
    format="%g"
)

rating = st.sidebar.select_slider(
    "Rating Makanan",
    options=[1, 2, 3, 4, 5],
    value=3
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
    st.subheader("Input Kandungan Nutrisi (per Porsi)")

    c1, c2, c3 = st.columns(3)

    with c1:
        kalori = st.number_input("Energi (Kalori)", min_value=0.0, step=0.1, format="%g")
        protein = st.number_input("Protein (g)", min_value=0.0, step=0.1, format="%g")
        serat = st.number_input("Serat (g)", min_value=0.0, step=0.1, format="%g")

    with c2:
        karbohidrat = st.number_input("Karbohidrat (g)", min_value=0.0, step=0.1, format="%g")
        lemak = st.number_input("Lemak (g)", min_value=0.0, step=0.1, format="%g")
        gula = st.number_input("Gula (g)", min_value=0.0, step=0.1, format="%g")

    with c3:
        natrium = st.number_input("Natrium (mg)", min_value=0.0, step=1.0, format="%g")
        kolesterol = st.number_input("Kolesterol (mg)", min_value=0.0, step=1.0, format="%g")
        porsi = st.number_input("Berat Porsi (g)", min_value=0.0, step=1.0, format="%g")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HASIL PREDIKSI
# =========================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Hasil Klasifikasi")

    if st.button("Proses Klasifikasi", use_container_width=True):

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
        st.caption("Ringkasan Informasi")
        st.write(f"**Nama Makanan:** {nama_makanan if nama_makanan else '-'}")
        st.write(f"**Asal Makanan:** {asal_makanan}")
        st.write(f"**Jenis Makanan:** {jenis_makanan}")
        st.write(f"**Waktu Persiapan:** {prep_time_min} menit")
        st.write(f"**Waktu Memasak:** {cook_time_min} menit")
        st.write(f"**Rating:**{rating}/5")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "Sistem ini menggunakan model Machine Learning berbasis Random Forest dan Gradient Boosting. "
    "Prediksi kesehatan makanan ditentukan berdasarkan kandungan nutrisi utama."
)
