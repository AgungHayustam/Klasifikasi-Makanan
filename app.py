import streamlit as st
import numpy as np
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Sistem Klasifikasi Makanan Sehat",
    page_icon="🥗",
    layout="wide"
)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
model = joblib.load("model_makanan.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# CUSTOM CSS – ACADEMIC STYLE
# ======================================================
st.markdown("""
<style>
.title {
    font-size: 34px;
    font-weight: 700;
    color: #2C2C2C;
}
.subtitle {
    font-size: 15px;
    color: #555555;
    margin-bottom: 25px;
}
.card {
    background-color: #FFFFFF;
    padding: 22px;
    border-radius: 12px;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.result-healthy {
    background-color: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 15px;
    border-radius: 6px;
    color: #2E7D32;
    font-weight: bold;
    text-align: center;
}
.result-unhealthy {
    background-color: #FDECEA;
    border-left: 6px solid #C62828;
    padding: 15px;
    border-radius: 6px;
    color: #C62828;
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
    ["Lokal", "Tradisional", "Internasional", "Cepat Saji", "Rumahan"]
)

jenis_makanan = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Makan Malam", "Dessert"]
)

prep_time = st.sidebar.number_input(
    "Waktu Persiapan (menit)",
    min_value=0,
    step=1
)

cook_time = st.sidebar.number_input(
    "Waktu Memasak (menit)",
    min_value=0,
    step=1
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
    st.subheader("Input Kandungan Nutrisi (per Porsi)")

    c1, c2, c3 = st.columns(3)

    with c1:
        kalori = st.number_input("Energi (Kalori)", min_value=0.0, step=1.0)
        protein = st.number_input("Protein (g)", min_value=0.0, step=0.1)
        serat = st.number_input("Serat (g)", min_value=0.0, step=0.1)

    with c2:
        karbohidrat = st.number_input("Karbohidrat (g)", min_value=0.0, step=0.1)
        lemak = st.number_input("Lemak (g)", min_value=0.0, step=0.1)
        gula = st.number_input("Gula (g)", min_value=0.0, step=0.1)

    with c3:
        natrium = st.number_input("Natrium (mg)", min_value=0.0, step=1.0)
        kolesterol = st.number_input("Kolesterol (mg)", min_value=0.0, step=1.0)
        porsi = st.number_input("Berat Porsi (g)", min_value=0.0, step=1.0)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDIKSI
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
            st.markdown('<div class="result-healthy">MAKANAN TERGOLONG SEHAT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-unhealthy">MAKANAN TERGOLONG TIDAK SEHAT</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("Informasi Tambahan")
        st.write(f"**Nama Makanan:** {nama_makanan if nama_makanan else '-'}")
        st.write(f"**Asal Makanan:** {asal_makanan}")
        st.write(f"**Jenis Makanan:** {jenis_makanan}")
        st.write(f"**Waktu Persiapan:** {prep_time} menit")
        st.write(f"**Waktu Memasak:** {cook_time} menit")
        st.write(f"**Rating:** ⭐ {rating}/5")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "Sistem ini menggunakan model Machine Learning berbasis Random Forest dan Gradient Boosting. "
    "Prediksi kesehatan makanan ditentukan berdasarkan atribut nutrisi, sedangkan informasi lainnya bersifat deskriptif."
)
