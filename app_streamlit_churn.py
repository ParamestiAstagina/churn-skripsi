import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Churn Nasabah Kartu Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .title-text {
        font-size: 36px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0px;
    }
    .subtitle-text {
        font-size: 17px;
        color: #475569;
        margin-top: 4px;
        margin-bottom: 22px;
    }
    .section-title {
        font-size: 25px;
        font-weight: 750;
        color: #0f172a;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .info-card {
        background: white;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        margin-bottom: 14px;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        text-align: center;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #1d4ed8;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        margin-top: 4px;
    }
    .risk-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }
    .risk-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 700;
        display: inline-block;
    }
    .prediction-box-churn {
        background: linear-gradient(135deg, #fee2e2 0%, #ffffff 100%);
        border: 1px solid #fecaca;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(185, 28, 28, 0.08);
    }
    .prediction-box-safe {
        background: linear-gradient(135deg, #dcfce7 0%, #ffffff 100%);
        border: 1px solid #bbf7d0;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(22, 101, 52, 0.08);
    }
    div[data-testid="stSidebar"] {
        background-color: #0f172a;
    }
    div[data-testid="stSidebar"] * {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# PATH FILE
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "credit_card_churn.csv"
MODEL_PATH = BASE_DIR / "model" / "catboost_bayes_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "model_metrics.pkl"
BEST_PARAMS_PATH = BASE_DIR / "model" / "best_params.pkl"

# =====================================================
# DAFTAR FITUR
# =====================================================
TARGET_COLUMN = "Attrition_Flag"
ID_COLUMNS = ["CLIENTNUM", "Customer_ID"]

FEATURE_COLUMNS = [
    "Customer_Age",
    "Gender",
    "Dependent_count",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

NUMERIC_COLUMNS = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]

FEATURE_DESCRIPTIONS = {
    "Customer_Age": "Usia nasabah",
    "Gender": "Jenis kelamin nasabah",
    "Dependent_count": "Jumlah tanggungan nasabah",
    "Education_Level": "Tingkat pendidikan nasabah",
    "Marital_Status": "Status pernikahan nasabah",
    "Income_Category": "Kategori pendapatan nasabah",
    "Card_Category": "Jenis kartu kredit",
    "Months_on_book": "Lama menjadi nasabah bank dalam bulan",
    "Total_Relationship_Count": "Jumlah produk atau relasi yang dimiliki nasabah",
    "Months_Inactive_12_mon": "Jumlah bulan tidak aktif dalam 12 bulan terakhir",
    "Contacts_Count_12_mon": "Jumlah kontak dengan bank dalam 12 bulan terakhir",
    "Credit_Limit": "Limit kartu kredit",
    "Total_Revolving_Bal": "Total saldo berjalan kartu kredit",
    "Avg_Open_To_Buy": "Rata-rata limit yang masih tersedia",
    "Total_Amt_Chng_Q4_Q1": "Perubahan nominal transaksi Q4 terhadap Q1",
    "Total_Trans_Amt": "Total nominal transaksi",
    "Total_Trans_Ct": "Total jumlah transaksi",
    "Total_Ct_Chng_Q4_Q1": "Perubahan jumlah transaksi Q4 terhadap Q1",
    "Avg_Utilization_Ratio": "Rata-rata rasio penggunaan kartu kredit",
}

# =====================================================
# FUNGSI UTILITAS
# =====================================================
@st.cache_data
def load_dataset():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_pickle(path):
    if path.exists():
        return joblib.load(path)
    return None


def show_title(title, subtitle=None):
    st.markdown(f"<div class='title-text'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle-text'>{subtitle}</div>", unsafe_allow_html=True)


def show_section(title):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)


def card(content):
    st.markdown(f"<div class='info-card'>{content}</div>", unsafe_allow_html=True)


def safe_percentage(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "-"


def normalize_target(y):
    """Mengubah target menjadi 1 = churn, 0 = tidak churn."""
    if y.dtype == "object":
        return y.astype(str).str.lower().apply(
            lambda x: 1 if "attrited" in x or "churn" in x or x == "1" else 0
        )
    return y.astype(int)


def get_prediction_proba(model, X):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba.ravel()


def get_risk_level(probability):
    if probability >= 0.70:
        return "Tinggi"
    elif probability >= 0.40:
        return "Sedang"
    return "Rendah"


def risk_badge(risk):
    if risk == "Tinggi":
        return "<span class='risk-high'>Risiko Tinggi</span>"
    if risk == "Sedang":
        return "<span class='risk-medium'>Risiko Sedang</span>"
    return "<span class='risk-low'>Risiko Rendah</span>"


def get_recommendation(risk, top_factors=None):
    top_factors = top_factors or []
    recommendations = []

    if risk == "Tinggi":
        recommendations.append("Hubungi nasabah secara personal untuk mengetahui kendala atau keluhan.")
        recommendations.append("Berikan penawaran retensi seperti cashback, reward point, atau promo transaksi.")
    elif risk == "Sedang":
        recommendations.append("Lakukan monitoring aktivitas nasabah dan berikan reminder penggunaan kartu.")
        recommendations.append("Tawarkan promo ringan untuk meningkatkan frekuensi transaksi.")
    else:
        recommendations.append("Pertahankan kualitas layanan dan lakukan monitoring berkala.")

    joined = " ".join(top_factors).lower()
    if "months_inactive" in joined:
        recommendations.append("Karena tingkat tidak aktif tinggi, berikan program aktivasi kembali.")
    if "total_trans_ct" in joined or "total_trans_amt" in joined:
        recommendations.append("Karena aktivitas transaksi rendah, tawarkan promo transaksi atau reward penggunaan kartu.")
    if "contacts_count" in joined:
        recommendations.append("Karena jumlah kontak dengan bank tinggi, evaluasi kemungkinan keluhan layanan.")
    if "utilization" in joined:
        recommendations.append("Evaluasi pola penggunaan limit dan sesuaikan penawaran produk dengan kebutuhan nasabah.")

    return recommendations


def validate_input_columns(df):
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    return missing_cols


def prepare_features(df):
    X = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FEATURE_COLUMNS]

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)
    return X


def make_template_csv():
    sample = pd.DataFrame([
        {
            "Customer_Age": 45,
            "Gender": "M",
            "Dependent_count": 3,
            "Education_Level": "Graduate",
            "Marital_Status": "Married",
            "Income_Category": "$60K - $80K",
            "Card_Category": "Blue",
            "Months_on_book": 39,
            "Total_Relationship_Count": 5,
            "Months_Inactive_12_mon": 1,
            "Contacts_Count_12_mon": 3,
            "Credit_Limit": 12691.0,
            "Total_Revolving_Bal": 777,
            "Avg_Open_To_Buy": 11914.0,
            "Total_Amt_Chng_Q4_Q1": 1.335,
            "Total_Trans_Amt": 1144,
            "Total_Trans_Ct": 42,
            "Total_Ct_Chng_Q4_Q1": 1.625,
            "Avg_Utilization_Ratio": 0.061,
        }
    ])
    return sample.to_csv(index=False).encode("utf-8")


def plot_confusion_matrix(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=["Tidak Churn", "Churn"],
        y=["Tidak Churn", "Churn"],
        aspect="auto",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return explainer, shap_values


def get_top_shap_factors(model, row_df, top_n=3):
    try:
        _, values = compute_shap_values(model, row_df)
        values = values[0]
        temp = pd.DataFrame({
            "Fitur": row_df.columns,
            "Nilai_SHAP": values,
            "Abs_SHAP": np.abs(values),
        }).sort_values("Abs_SHAP", ascending=False)
        return temp.head(top_n)
    except Exception:
        return pd.DataFrame(columns=["Fitur", "Nilai_SHAP", "Abs_SHAP"])


# =====================================================
# LOAD DATA DAN MODEL
# =====================================================
df = load_dataset()
model = load_model()
metrics_data = load_pickle(METRICS_PATH)
best_params = load_pickle(BEST_PARAMS_PATH)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("# 💳 Churn Prediction")
st.sidebar.markdown("CatBoost + Bayesian Optimization + SHAP")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Dashboard",
        "Dataset & EDA",
        "Model Final",
        "Evaluasi Model",
        "Interpretasi SHAP",
        "Prediksi Churn",
        "Rekomendasi Retensi",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Sistem ini menggunakan model CatBoost hasil Optimasi Bayesian untuk memprediksi churn nasabah kartu kredit."
)

# =====================================================
# WARNING JIKA FILE BELUM ADA
# =====================================================
if model is None:
    st.warning(
        "Model belum ditemukan. Simpan model final kamu pada path: `model/catboost_bayes_model.pkl`. "
        "Beberapa fitur prediksi tidak akan berjalan sebelum model tersedia."
    )

if df is None:
    st.warning(
        "Dataset belum ditemukan. Simpan dataset pada path: `data/credit_card_churn.csv`. "
        "Beberapa halaman EDA dan evaluasi tidak akan menampilkan data."
    )

# =====================================================
# HALAMAN DASHBOARD
# =====================================================
if menu == "Dashboard":
    show_title(
        "Prediksi Churn Nasabah Kartu Kredit",
        "Implementasi sistem machine learning menggunakan CatBoost dengan Optimasi Bayesian dan Interpretasi SHAP.",
    )

    col1, col2, col3, col4 = st.columns(4)
    total_data = len(df) if df is not None else 0
    total_features = len(FEATURE_COLUMNS)
    churn_rate = "-"
    auc_value = "-"

    if df is not None and TARGET_COLUMN in df.columns:
        y_norm = normalize_target(df[TARGET_COLUMN])
        churn_rate = f"{y_norm.mean() * 100:.2f}%"

    if metrics_data and "roc_auc" in metrics_data:
        auc_value = f"{metrics_data['roc_auc']:.4f}"

    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{total_data}</div><div class='metric-label'>Jumlah Data</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{total_features}</div><div class='metric-label'>Jumlah Fitur</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{churn_rate}</div><div class='metric-label'>Persentase Churn</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{auc_value}</div><div class='metric-label'>ROC-AUC Model</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        card(
            """
            <h3>Tujuan Sistem</h3>
            <p>
            Sistem ini dirancang untuk membantu pihak bank mengidentifikasi nasabah kartu kredit
            yang berpotensi melakukan churn. Hasil prediksi dilengkapi probabilitas risiko,
            interpretasi faktor penyebab menggunakan SHAP, dan rekomendasi retensi nasabah.
            </p>
            """
        )
        card(
            """
            <h3>Alur Sistem</h3>
            <p>
            Dataset → Preprocessing → Bayesian Optimization → CatBoost Final → Evaluasi Model →
            Interpretasi SHAP → Prediksi Churn → Rekomendasi Retensi
            </p>
            """
        )

    with col_right:
        method_df = pd.DataFrame(
            {
                "Komponen": [
                    "Algoritma",
                    "Optimasi",
                    "Interpretasi",
                    "Output",
                    "Penggunaan",
                ],
                "Keterangan": [
                    "CatBoost Classifier",
                    "Bayesian Optimization",
                    "SHAP",
                    "Churn / Tidak Churn",
                    "Input manual dan upload CSV",
                ],
            }
        )
        st.dataframe(method_df, use_container_width=True, hide_index=True)

# =====================================================
# HALAMAN DATASET & EDA
# =====================================================
elif menu == "Dataset & EDA":
    show_title("Dataset & Exploratory Data Analysis", "Menampilkan informasi dataset, distribusi target, dan pola awal data nasabah.")

    if df is None:
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Informasi Dataset", "Visualisasi Target", "Analisis Fitur"])

    with tab1:
        show_section("Preview Dataset")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", df.shape[0])
        col2.metric("Jumlah Kolom", df.shape[1])
        col3.metric("Missing Value", int(df.isnull().sum().sum()))

        show_section("Deskripsi Fitur")
        desc_df = pd.DataFrame(
            [{"Fitur": k, "Keterangan": v} for k, v in FEATURE_DESCRIPTIONS.items()]
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)

    with tab2:
        if TARGET_COLUMN not in df.columns:
            st.error(f"Kolom target `{TARGET_COLUMN}` tidak ditemukan pada dataset.")
        else:
            target_count = df[TARGET_COLUMN].value_counts().reset_index()
            target_count.columns = ["Status", "Jumlah"]

            col1, col2 = st.columns([1, 1])
            with col1:
                fig = px.bar(target_count, x="Status", y="Jumlah", text="Jumlah", title="Distribusi Status Churn")
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.pie(target_count, names="Status", values="Jumlah", title="Proporsi Status Churn", hole=0.45)
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)

            st.info(
                "Distribusi target digunakan untuk melihat apakah data churn dan tidak churn seimbang. "
                "Pada kasus churn, data sering kali tidak seimbang sehingga evaluasi tidak cukup hanya menggunakan accuracy."
            )

    with tab3:
        selected_num = st.selectbox("Pilih fitur numerik", NUMERIC_COLUMNS)
        fig = px.histogram(
            df,
            x=selected_num,
            color=TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
            marginal="box",
            title=f"Distribusi {selected_num}",
        )
        st.plotly_chart(fig, use_container_width=True)

        selected_cat = st.selectbox("Pilih fitur kategorikal", CATEGORICAL_COLUMNS)
        if TARGET_COLUMN in df.columns:
            grouped = df.groupby([selected_cat, TARGET_COLUMN]).size().reset_index(name="Jumlah")
            fig = px.bar(
                grouped,
                x=selected_cat,
                y="Jumlah",
                color=TARGET_COLUMN,
                barmode="group",
                title=f"Perbandingan Churn berdasarkan {selected_cat}",
            )
            st.plotly_chart(fig, use_container_width=True)

        show_section("Korelasi Fitur Numerik")
        numeric_available = [c for c in NUMERIC_COLUMNS if c in df.columns]
        corr = df[numeric_available].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=False, aspect="auto", title="Heatmap Korelasi")
        fig.update_layout(height=650)
        st.plotly_chart(fig, use_container_width=True)

# =====================================================
# HALAMAN MODEL FINAL
# =====================================================
elif menu == "Model Final":
    show_title("Model Final CatBoost Optimasi Bayesian", "Model yang digunakan pada sistem adalah CatBoost dengan hyperparameter terbaik hasil Bayesian Optimization.")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        card(
            """
            <h3>Deskripsi Model</h3>
            <p>
            CatBoost Classifier digunakan untuk memprediksi status churn nasabah kartu kredit.
            Model final yang digunakan pada sistem ini merupakan model yang telah dioptimasi
            menggunakan Bayesian Optimization sehingga hyperparameter yang dipakai adalah
            kombinasi terbaik dari proses pencarian parameter.
            </p>
            """
        )
        card(
            """
            <h3>Alur Pemodelan</h3>
            <p>
            Dataset → Pemilihan Fitur → Preprocessing → Bayesian Optimization →
            CatBoost Final → Evaluasi → Penyimpanan Model → Implementasi Web Streamlit
            </p>
            """
        )

    with col2:
        st.markdown("### Fitur Model")
        st.dataframe(pd.DataFrame({"Fitur": FEATURE_COLUMNS}), use_container_width=True, hide_index=True)

    show_section("Hyperparameter Terbaik")
    if best_params:
        params_df = pd.DataFrame(
            [{"Hyperparameter": k, "Nilai Terbaik": v} for k, v in best_params.items()]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "File `best_params.pkl` belum ditemukan. Jika ingin menampilkan parameter terbaik, "
            "simpan dictionary hasil Bayesian Optimization ke `model/best_params.pkl`."
        )
        sample_params = pd.DataFrame(
            {
                "Hyperparameter": [
                    "iterations",
                    "depth",
                    "learning_rate",
                    "l2_leaf_reg",
                    "bagging_temperature",
                    "random_strength",
                ],
                "Nilai Terbaik": ["...", "...", "...", "...", "...", "..."],
            }
        )
        st.dataframe(sample_params, use_container_width=True, hide_index=True)

# =====================================================
# HALAMAN EVALUASI MODEL
# =====================================================
elif menu == "Evaluasi Model":
    show_title("Evaluasi Model", "Menampilkan performa model final CatBoost hasil Optimasi Bayesian.")

    if df is None or model is None:
        st.stop()

    if TARGET_COLUMN not in df.columns:
        st.error(f"Kolom target `{TARGET_COLUMN}` tidak ditemukan pada dataset.")
        st.stop()

    X = prepare_features(df)
    y = normalize_target(df[TARGET_COLUMN])

    try:
        y_proba = get_prediction_proba(model, X)
        y_pred = (y_proba >= 0.5).astype(int)

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_proba)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1-Score", f"{f1:.4f}")
        col5.metric("ROC-AUC", f"{auc:.4f}")

        col_left, col_right = st.columns([1, 1])
        with col_left:
            show_section("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

        with col_right:
            show_section("ROC Curve")
            fpr, tpr, _ = roc_curve(y, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.4f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Model", line=dict(dash="dash")))
            fig.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        show_section("Classification Report")
        report = classification_report(y, y_pred, target_names=["Tidak Churn", "Churn"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.info(
            "Pada kasus churn, recall kelas churn penting diperhatikan karena menunjukkan kemampuan model "
            "mendeteksi nasabah yang benar-benar berisiko churn."
        )

    except Exception as e:
        st.error(f"Evaluasi gagal dilakukan: {e}")

# =====================================================
# HALAMAN INTERPRETASI SHAP
# =====================================================
elif menu == "Interpretasi SHAP":
    show_title("Interpretasi SHAP", "Menjelaskan faktor yang memengaruhi prediksi churn secara global dan lokal.")

    if df is None or model is None:
        st.stop()

    X = prepare_features(df)
    sample_size = st.slider("Jumlah sampel untuk interpretasi SHAP", 50, min(1000, len(X)), min(300, len(X)), step=50)
    X_sample = X.sample(sample_size, random_state=42)

    tab1, tab2 = st.tabs(["Interpretasi Global", "Interpretasi Lokal"])

    with tab1:
        st.info(
            "Nilai SHAP menunjukkan kontribusi setiap fitur terhadap output model. "
            "Semakin besar nilai absolut SHAP, semakin besar pengaruh fitur terhadap prediksi."
        )

        try:
            explainer, shap_values = compute_shap_values(model, X_sample)
            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.DataFrame({
                "Fitur": X_sample.columns,
                "Mean Absolute SHAP": mean_abs,
            }).sort_values("Mean Absolute SHAP", ascending=False)

            fig = px.bar(
                shap_importance.head(15),
                x="Mean Absolute SHAP",
                y="Fitur",
                orientation="h",
                title="Top 15 Fitur Paling Berpengaruh berdasarkan SHAP",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=560)
            st.plotly_chart(fig, use_container_width=True)

            show_section("Tabel Ranking SHAP")
            st.dataframe(shap_importance, use_container_width=True, hide_index=True)

            show_section("SHAP Summary Plot")
            fig_shap, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig_shap, bbox_inches="tight")
            plt.close(fig_shap)

        except Exception as e:
            st.error(f"SHAP gagal dihitung: {e}")

    with tab2:
        if ID_COLUMNS[0] in df.columns:
            id_col = ID_COLUMNS[0]
        elif ID_COLUMNS[1] in df.columns:
            id_col = ID_COLUMNS[1]
        else:
            id_col = None

        row_index = st.number_input("Pilih index data nasabah", min_value=0, max_value=len(df) - 1, value=0)
        row = X.iloc[[row_index]]

        try:
            probability = float(get_prediction_proba(model, row)[0])
            prediction = "Churn" if probability >= 0.5 else "Tidak Churn"
            risk = get_risk_level(probability)

            col1, col2, col3 = st.columns(3)
            col1.metric("Prediksi", prediction)
            col2.metric("Probabilitas Churn", f"{probability * 100:.2f}%")
            col3.markdown(risk_badge(risk), unsafe_allow_html=True)

            show_section("Data Nasabah Terpilih")
            st.dataframe(row, use_container_width=True)

            top_factors = get_top_shap_factors(model, row, top_n=8)
            if not top_factors.empty:
                show_section("Faktor Utama berdasarkan SHAP")
                fig = px.bar(
                    top_factors.sort_values("Abs_SHAP"),
                    x="Nilai_SHAP",
                    y="Fitur",
                    orientation="h",
                    title="Kontribusi Fitur pada Prediksi Nasabah Ini",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(top_factors, use_container_width=True, hide_index=True)

                positive = top_factors[top_factors["Nilai_SHAP"] > 0]["Fitur"].head(3).tolist()
                negative = top_factors[top_factors["Nilai_SHAP"] < 0]["Fitur"].head(3).tolist()

                if positive:
                    st.error("Faktor yang mendorong prediksi ke arah churn: " + ", ".join(positive))
                if negative:
                    st.success("Faktor yang menekan risiko churn: " + ", ".join(negative))

        except Exception as e:
            st.error(f"Interpretasi lokal gagal dilakukan: {e}")

# =====================================================
# HALAMAN PREDIKSI CHURN
# =====================================================
elif menu == "Prediksi Churn":
    show_title("Prediksi Churn", "Lakukan prediksi melalui input manual atau upload file CSV.")

    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

    with tab1:
        st.markdown("### Masukkan Data Nasabah")

        with st.form("manual_prediction_form"):
            st.markdown("#### Data Demografis")
            c1, c2, c3 = st.columns(3)
            with c1:
                customer_age = st.number_input("Usia Nasabah", 18, 100, 45)
                gender = st.selectbox("Jenis Kelamin", ["M", "F"])
            with c2:
                dependent_count = st.number_input("Jumlah Tanggungan", 0, 10, 2)
                education_level = st.selectbox(
                    "Tingkat Pendidikan",
                    ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"],
                )
            with c3:
                marital_status = st.selectbox("Status Pernikahan", ["Unknown", "Single", "Married", "Divorced"])
                income_category = st.selectbox(
                    "Kategori Pendapatan",
                    ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"],
                )

            st.markdown("#### Data Kartu Kredit")
            c1, c2, c3 = st.columns(3)
            with c1:
                card_category = st.selectbox("Jenis Kartu", ["Blue", "Silver", "Gold", "Platinum"])
                months_on_book = st.number_input("Lama Menjadi Nasabah", 1, 80, 36)
            with c2:
                total_relationship_count = st.number_input("Total Relationship Count", 1, 10, 4)
                credit_limit = st.number_input("Credit Limit", 0.0, 50000.0, 5000.0, step=100.0)
            with c3:
                total_revolving_bal = st.number_input("Total Revolving Balance", 0.0, 5000.0, 1000.0, step=50.0)
                avg_open_to_buy = st.number_input("Avg Open To Buy", 0.0, 50000.0, 4000.0, step=100.0)

            st.markdown("#### Data Aktivitas Transaksi")
            c1, c2, c3 = st.columns(3)
            with c1:
                months_inactive = st.number_input("Months Inactive 12 Bulan", 0, 12, 2)
                contacts_count = st.number_input("Contacts Count 12 Bulan", 0, 12, 2)
            with c2:
                total_amt_chng = st.number_input("Total Amt Chng Q4/Q1", 0.0, 5.0, 0.75, step=0.01)
                total_trans_amt = st.number_input("Total Trans Amt", 0.0, 25000.0, 4000.0, step=100.0)
            with c3:
                total_trans_ct = st.number_input("Total Trans Ct", 0, 200, 60)
                total_ct_chng = st.number_input("Total Ct Chng Q4/Q1", 0.0, 5.0, 0.70, step=0.01)
                avg_utilization = st.number_input("Avg Utilization Ratio", 0.0, 1.0, 0.30, step=0.01)

            submitted = st.form_submit_button("Prediksi Churn", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([
                {
                    "Customer_Age": customer_age,
                    "Gender": gender,
                    "Dependent_count": dependent_count,
                    "Education_Level": education_level,
                    "Marital_Status": marital_status,
                    "Income_Category": income_category,
                    "Card_Category": card_category,
                    "Months_on_book": months_on_book,
                    "Total_Relationship_Count": total_relationship_count,
                    "Months_Inactive_12_mon": months_inactive,
                    "Contacts_Count_12_mon": contacts_count,
                    "Credit_Limit": credit_limit,
                    "Total_Revolving_Bal": total_revolving_bal,
                    "Avg_Open_To_Buy": avg_open_to_buy,
                    "Total_Amt_Chng_Q4_Q1": total_amt_chng,
                    "Total_Trans_Amt": total_trans_amt,
                    "Total_Trans_Ct": total_trans_ct,
                    "Total_Ct_Chng_Q4_Q1": total_ct_chng,
                    "Avg_Utilization_Ratio": avg_utilization,
                }
            ])

            X_input = prepare_features(input_df)
            probability = float(get_prediction_proba(model, X_input)[0])
            prediction = "Churn" if probability >= 0.5 else "Tidak Churn"
            risk = get_risk_level(probability)
            top_factors = get_top_shap_factors(model, X_input, top_n=5)
            top_factor_list = top_factors["Fitur"].tolist() if not top_factors.empty else []
            recs = get_recommendation(risk, top_factor_list)

            box_class = "prediction-box-churn" if prediction == "Churn" else "prediction-box-safe"
            st.markdown(
                f"""
                <div class='{box_class}'>
                    <h2>Hasil Prediksi: {prediction}</h2>
                    <h3>Probabilitas Churn: {probability * 100:.2f}%</h3>
                    {risk_badge(risk)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_left, col_right = st.columns([1, 1])
            with col_left:
                show_section("Faktor Utama")
                if not top_factors.empty:
                    st.dataframe(top_factors, use_container_width=True, hide_index=True)
                    fig = px.bar(top_factors, x="Abs_SHAP", y="Fitur", orientation="h")
                    fig.update_layout(yaxis=dict(autorange="reversed"), height=360)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Faktor utama belum dapat dihitung.")

            with col_right:
                show_section("Rekomendasi Retensi")
                for rec in recs:
                    st.write(f"- {rec}")

    with tab2:
        st.markdown("### Upload Data Nasabah")
        st.write("Upload file CSV dengan format kolom yang sama seperti fitur model.")

        st.download_button(
            label="Download Template CSV",
            data=make_template_csv(),
            file_name="template_prediksi_churn.csv",
            mime="text/csv",
            use_container_width=True,
        )

        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

        if uploaded_file is not None:
            uploaded_df = pd.read_csv(uploaded_file)
            st.markdown("#### Preview Data Upload")
            st.dataframe(uploaded_df.head(20), use_container_width=True)

            missing_cols = validate_input_columns(uploaded_df)
            if missing_cols:
                st.error("Kolom berikut belum ada pada file CSV: " + ", ".join(missing_cols))
            else:
                if st.button("Proses Prediksi Batch", use_container_width=True):
                    X_batch = prepare_features(uploaded_df)
                    probabilities = get_prediction_proba(model, X_batch)
                    predictions = np.where(probabilities >= 0.5, "Churn", "Tidak Churn")
                    risks = [get_risk_level(p) for p in probabilities]

                    result_df = uploaded_df.copy()
                    result_df["Prediksi"] = predictions
                    result_df["Probabilitas_Churn"] = probabilities
                    result_df["Probabilitas_Churn_Persen"] = [f"{p * 100:.2f}%" for p in probabilities]
                    result_df["Risiko"] = risks

                    faktor_utama_list = []
                    rekomendasi_list = []
                    max_explain = min(50, len(X_batch))

                    for i in range(len(X_batch)):
                        if i < max_explain:
                            row = X_batch.iloc[[i]]
                            top_factors = get_top_shap_factors(model, row, top_n=3)
                            factors = top_factors["Fitur"].tolist() if not top_factors.empty else []
                        else:
                            factors = []

                        faktor_utama_list.append(", ".join(factors) if factors else "-"
                        )
                        rekomendasi_list.append("; ".join(get_recommendation(risks[i], factors)))

                    result_df["Faktor_Utama"] = faktor_utama_list
                    result_df["Rekomendasi"] = rekomendasi_list

                    st.success("Prediksi batch berhasil dilakukan.")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Data", len(result_df))
                    col2.metric("Prediksi Churn", int((result_df["Prediksi"] == "Churn").sum()))
                    col3.metric("Prediksi Tidak Churn", int((result_df["Prediksi"] == "Tidak Churn").sum()))

                    fig = px.histogram(result_df, x="Risiko", color="Risiko", title="Distribusi Tingkat Risiko")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Hasil Prediksi")
                    st.dataframe(result_df, use_container_width=True)

                    csv_result = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Hasil Prediksi CSV",
                        data=csv_result,
                        file_name="hasil_prediksi_churn.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

# =====================================================
# HALAMAN REKOMENDASI RETENSI
# =====================================================
elif menu == "Rekomendasi Retensi":
    show_title("Rekomendasi Retensi Nasabah", "Strategi retensi berdasarkan tingkat risiko churn dan faktor penyebab utama.")

    st.markdown("### Matriks Rekomendasi")
    rec_df = pd.DataFrame(
        {
            "Tingkat Risiko": ["Rendah", "Sedang", "Tinggi"],
            "Kondisi": [
                "Probabilitas churn kurang dari 40%",
                "Probabilitas churn antara 40% sampai 69%",
                "Probabilitas churn minimal 70%",
            ],
            "Rekomendasi": [
                "Pertahankan layanan dan lakukan monitoring berkala.",
                "Berikan reminder penggunaan kartu, promo ringan, atau penawaran reward.",
                "Hubungi nasabah secara personal, identifikasi keluhan, dan berikan penawaran retensi khusus.",
            ],
        }
    )
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown("### Rekomendasi Berdasarkan Faktor SHAP")
    shap_rec_df = pd.DataFrame(
        {
            "Faktor Utama": [
                "Months_Inactive_12_mon tinggi",
                "Total_Trans_Ct rendah",
                "Total_Trans_Amt rendah",
                "Contacts_Count_12_mon tinggi",
                "Avg_Utilization_Ratio tidak optimal",
                "Total_Relationship_Count rendah",
            ],
            "Interpretasi": [
                "Nasabah jarang atau tidak aktif menggunakan kartu.",
                "Frekuensi transaksi nasabah rendah.",
                "Nominal transaksi nasabah rendah.",
                "Nasabah sering menghubungi bank, kemungkinan ada kendala atau keluhan.",
                "Pola penggunaan limit kartu kurang ideal.",
                "Nasabah hanya memiliki sedikit produk atau relasi dengan bank.",
            ],
            "Rekomendasi Retensi": [
                "Kirim program aktivasi kembali, reminder, atau bonus transaksi pertama.",
                "Tawarkan cashback, reward point, atau promo transaksi rutin.",
                "Berikan penawaran transaksi bernilai tinggi seperti cicilan ringan atau promo merchant.",
                "Evaluasi kualitas layanan dan lakukan follow-up keluhan nasabah.",
                "Tinjau ulang limit, edukasi penggunaan kartu, atau tawarkan produk yang lebih sesuai.",
                "Tawarkan bundling produk, upgrade kartu, atau benefit tambahan agar loyalitas meningkat.",
            ],
        }
    )
    st.dataframe(shap_rec_df, use_container_width=True, hide_index=True)

    card(
        """
        <h3>Catatan Implementasi</h3>
        <p>
        Rekomendasi pada sistem ini bersifat rule-based yang memanfaatkan tingkat risiko churn
        dan faktor utama dari interpretasi SHAP. Rekomendasi dapat dikembangkan lebih lanjut
        sesuai kebijakan bisnis bank.
        </p>
        """
    )
