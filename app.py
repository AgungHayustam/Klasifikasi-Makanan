import streamlit as st
import numpy as np
import joblib

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Klasifikasi Makanan Sehat",
    page_icon="🥗",
    layout="wide"
)

# ======================================================
# CUSTOM CSS (MINIMAL & MODERN)
# ======================================================
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #2E7D32;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
    }
    .card {
        padding: 25px;
        border-radius: 12px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .result-healthy {
        background-color: #E8F5E9;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        color: #2E7D32;
        font-weight: bold;
        text-align: center;
    }
    .result-unhealthy {
        background-color: #FDECEA;
        padding: 20px;
        border-radius: 10px;
        font-size: 20px;
        color: #C62828;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD MODEL & SCALER
# ======================================================
model = joblib.load("model_makanan.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# HEADER
# ======================================================
st.markdown('<div class="main-title">🥗 Sistem Klasifikasi Makanan Sehat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Prediksi makanan sehat & tidak sehat berbasis kandungan nutrisi menggunakan Machine Learning</div>', unsafe_allow_html=True)

st.divider()

# ======================================================
# SIDEBAR - INFORMASI UMUM
# ======================================================
st.sidebar.header("📝 Informasi Makanan")

nama_makanan = st.sidebar.text_input("Nama Makanan")

asal_masakan = st.sidebar.selectbox(
    "Asal Masakan",
    ["Indonesia", "Asia", "Eropa", "Amerika", "Timur Tengah"]
)

jenis_makanan = st.sidebar.selectbox(
    "Jenis Makanan",
    ["Makanan Utama", "Sarapan", "Cemilan", "Dinner", "Dessert"]
)

jenis_diet = st.sidebar.selectbox(
    "Jenis Diet",
    ["Umum", "Vegetarian", "Vegan", "Rendah Lemak", "Rendah Gula", "Tinggi Protein"]
)

metode_masak = st.sidebar.selectbox(
    "Metode Memasak",
    ["Rebus", "Kukus", "Panggang", "Tumis", "Goreng", "Tanpa Dimasak"]
)

waktu_persiapan = st.sidebar.number_input("Waktu Persiapan (menit)", min_value=0)
waktu_memasak = st.sidebar.number_input("Waktu Memasak (menit)", min_value=0)

rating = st.sidebar.slider("Rating Makanan", 1, 5, 3)

# ======================================================
# MAIN CONTENT
# ======================================================
col_left, col_right = st.columns([2, 1])

# =========================
# INPUT NUTRISI
# =========================
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔢 Kandungan Nutrisi")

    c1, c2, c3 = st.columns(3)

    with c1:
        kalori = st.number_input("Kalori", min_value=0.0)
        protein = st.number_input("Protein (g)", min_value=0.0)
        serat = st.number_input("Serat (g)", min_value=0.0)

    with c2:
        karbohidrat = st.number_input("Karbohidrat (g)", min_value=0.0)
        lemak = st.number_input("Lemak (g)", min_value=0.0)
        gula = st.number_input("Gula (g)", min_value=0.0)

    with c3:
        natrium = st.number_input("Natrium (mg)", min_value=0.0)
        kolesterol = st.number_input("Kolesterol (mg)", min_value=0.0)
        porsi = st.number_input("Porsi Sajian (g)", min_value=0.0)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDIKSI & RINGKASAN
# =========================
with col_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Prediksi")

    if st.button("Prediksi Kesehatan", use_container_width=True):

        data_input = np.array([[ 
            kalori, protein, karbohidrat, lemak,
            serat, gula, natrium, kolesterol, porsi
        ]])

        data_scaled = scaler.transform(data_input)
        hasil_prediksi = model.predict(data_scaled)

        if hasil_prediksi[0] == 1:
            st.markdown('<div class="result-healthy">✅ MAKANAN SEHAT</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-unhealthy">❌ MAKANAN TIDAK SEHAT</div>', unsafe_allow_html=True)

        st.divider()
        st.caption("📋 Ringkasan")
        st.write(f"**Nama:** {nama_makanan if nama_makanan else '-'}")
        st.write(f"**Jenis:** {jenis_makanan}")
        st.write(f"**Diet:** {jenis_diet}")
        st.write(f"**Metode:** {metode_masak}")
        st.write(f"**Waktu Total:** {waktu_persiapan + waktu_memasak} menit")
        st.write(f"**Rating:** ⭐ {rating}/5")

    st.markdown('</div>', unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.divider()
st.caption(
    "🔬 Model Machine Learning berbasis Random Forest / Gradient Boosting. "
    "Prediksi hanya menggunakan data nutrisi, atribut lain bersifat informatif."
)
