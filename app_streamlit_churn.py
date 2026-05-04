import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Prediksi Churn Kartu Kredit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# STYLE
# =====================================================
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .title-text {font-size: 36px; font-weight: 800; color: #0f172a; margin-bottom: 0px;}
    .subtitle-text {font-size: 16px; color: #475569; margin-top: 4px; margin-bottom: 20px;}
    .section-title {font-size: 24px; font-weight: 750; color: #0f172a; margin-top: 18px; margin-bottom: 10px;}
    .card {background: #ffffff; padding: 22px; border-radius: 18px; border: 1px solid #e2e8f0; box-shadow: 0 8px 24px rgba(15,23,42,.06); margin-bottom: 14px;}
    .metric-card {background: linear-gradient(135deg,#fff,#f1f5f9); padding: 18px; border-radius: 18px; border: 1px solid #e2e8f0; text-align:center; box-shadow: 0 8px 24px rgba(15,23,42,.06);}
    .metric-value {font-size: 30px; font-weight: 800; color: #1d4ed8;}
    .metric-label {font-size: 14px; color: #64748b;}
    .risk-low {background:#dcfce7;color:#166534;padding:9px 14px;border-radius:999px;font-weight:800;display:inline-block;}
    .risk-medium {background:#fef9c3;color:#854d0e;padding:9px 14px;border-radius:999px;font-weight:800;display:inline-block;}
    .risk-high {background:#fee2e2;color:#991b1b;padding:9px 14px;border-radius:999px;font-weight:800;display:inline-block;}
    .pred-churn {background:linear-gradient(135deg,#fee2e2,#fff);border:1px solid #fecaca;border-radius:20px;padding:24px;box-shadow:0 8px 24px rgba(185,28,28,.08);}
    .pred-safe {background:linear-gradient(135deg,#dcfce7,#fff);border:1px solid #bbf7d0;border-radius:20px;padding:24px;box-shadow:0 8px 24px rgba(22,101,52,.08);}
    div[data-testid="stSidebar"] {background-color:#0f172a;}
    div[data-testid="stSidebar"] * {color:white;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# PATH DAN KOLOM
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_CANDIDATES = [
    BASE_DIR / "data" / "credit_card_churn.csv",
    BASE_DIR / "data" / "BankChurners.csv",
    BASE_DIR / "credit_card_churn.csv",
    BASE_DIR / "BankChurners.csv",
]

MODEL_CANDIDATES = [
    BASE_DIR / "model" / "catboost_bayes_model.pkl",
    BASE_DIR / "model" / "catboost_bayes_model.joblib",
    BASE_DIR / "model" / "best_model.pkl",
    BASE_DIR / "model" / "catboost_model.pkl",
    BASE_DIR / "catboost_bayes_model.pkl",
    BASE_DIR / "best_model.pkl",
    BASE_DIR / "catboost_model.pkl",
]

BEST_PARAMS_CANDIDATES = [BASE_DIR / "model" / "best_params.pkl", BASE_DIR / "best_params.pkl"]
METRICS_CANDIDATES = [BASE_DIR / "model" / "model_metrics.pkl", BASE_DIR / "model_metrics.pkl"]

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

NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]

FEATURE_DESCRIPTIONS = {
    "Customer_Age": "Usia nasabah",
    "Gender": "Jenis kelamin nasabah",
    "Dependent_count": "Jumlah tanggungan nasabah",
    "Education_Level": "Tingkat pendidikan nasabah",
    "Marital_Status": "Status pernikahan nasabah",
    "Income_Category": "Kategori pendapatan nasabah",
    "Card_Category": "Jenis kartu kredit",
    "Months_on_book": "Lama menjadi nasabah dalam bulan",
    "Total_Relationship_Count": "Jumlah produk/relasi dengan bank",
    "Months_Inactive_12_mon": "Jumlah bulan tidak aktif dalam 12 bulan terakhir",
    "Contacts_Count_12_mon": "Jumlah kontak dengan bank dalam 12 bulan terakhir",
    "Credit_Limit": "Limit kartu kredit",
    "Total_Revolving_Bal": "Total saldo berjalan",
    "Avg_Open_To_Buy": "Rata-rata limit tersedia",
    "Total_Amt_Chng_Q4_Q1": "Perubahan nominal transaksi Q4/Q1",
    "Total_Trans_Amt": "Total nominal transaksi",
    "Total_Trans_Ct": "Total jumlah transaksi",
    "Total_Ct_Chng_Q4_Q1": "Perubahan jumlah transaksi Q4/Q1",
    "Avg_Utilization_Ratio": "Rasio penggunaan kartu kredit",
}

# =====================================================
# HELPER
# =====================================================
def find_existing_file(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_data
def load_dataset_from_path(path_str):
    return pd.read_csv(path_str)


@st.cache_resource
def load_model_from_path(path_str):
    path = Path(path_str)
    if path.suffix.lower() == ".cbm":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost belum terinstall, tidak bisa membaca file .cbm")
        m = CatBoostClassifier()
        m.load_model(str(path))
        return m
    return joblib.load(path)


def load_model_from_upload(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        if suffix == ".cbm":
            if CatBoostClassifier is None:
                raise RuntimeError("catboost belum terinstall, tidak bisa membaca file .cbm")
            m = CatBoostClassifier()
            m.load_model(tmp_path)
            return m
        return joblib.load(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def load_pickle_optional(candidates):
    path = find_existing_file(candidates)
    if path is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def title(text, sub=None):
    st.markdown(f"<div class='title-text'>{text}</div>", unsafe_allow_html=True)
    if sub:
        st.markdown(f"<div class='subtitle-text'>{sub}</div>", unsafe_allow_html=True)


def section(text):
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def html_card(content):
    st.markdown(f"<div class='card'>{content}</div>", unsafe_allow_html=True)


def normalize_target(y):
    if y.dtype == "object":
        y_text = y.astype(str).str.lower()
        return y_text.apply(lambda x: 1 if "attrited" in x or "churn" in x or x == "1" else 0).astype(int)
    return y.astype(int)


def prepare_features(data):
    X = data.copy()
    for col in FEATURE_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan
    X = X[FEATURE_COLUMNS]
    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].astype(str)
    return X


def get_prediction_proba(model, X):
    proba = model.predict_proba(X)
    if len(proba.shape) == 2 and proba.shape[1] > 1:
        return proba[:, 1]
    return np.asarray(proba).ravel()


def risk_level(p):
    if p >= 0.70:
        return "Tinggi"
    if p >= 0.40:
        return "Sedang"
    return "Rendah"


def risk_badge(risk):
    if risk == "Tinggi":
        return "<span class='risk-high'>Risiko Tinggi</span>"
    if risk == "Sedang":
        return "<span class='risk-medium'>Risiko Sedang</span>"
    return "<span class='risk-low'>Risiko Rendah</span>"


def validate_columns(data):
    return [c for c in FEATURE_COLUMNS if c not in data.columns]


def recommendations(risk, factors=None):
    factors = factors or []
    text = " ".join(factors).lower()
    recs = []

    if risk == "Tinggi":
        recs += [
            "Hubungi nasabah secara personal untuk mengetahui kendala atau keluhan.",
            "Berikan penawaran retensi seperti cashback, reward point, atau promo transaksi.",
        ]
    elif risk == "Sedang":
        recs += [
            "Lakukan monitoring aktivitas nasabah.",
            "Berikan reminder penggunaan kartu atau promo ringan.",
        ]
    else:
        recs += ["Pertahankan kualitas layanan dan lakukan monitoring berkala."]

    if "months_inactive" in text:
        recs.append("Karena tingkat tidak aktif tinggi, berikan program aktivasi kembali.")
    if "total_trans_ct" in text or "total_trans_amt" in text:
        recs.append("Karena aktivitas transaksi rendah, tawarkan promo transaksi atau reward penggunaan kartu.")
    if "contacts_count" in text:
        recs.append("Karena jumlah kontak tinggi, evaluasi kemungkinan keluhan layanan.")
    if "utilization" in text:
        recs.append("Evaluasi pola penggunaan limit kartu dan sesuaikan penawaran produk.")
    return recs


def template_csv():
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


def top_shap_factors(model, X_row, top_n=5):
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X_row)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)[0]
        out = pd.DataFrame({"Fitur": X_row.columns, "Nilai_SHAP": values, "Abs_SHAP": np.abs(values)})
        return out.sort_values("Abs_SHAP", ascending=False).head(top_n)
    except Exception:
        return pd.DataFrame(columns=["Fitur", "Nilai_SHAP", "Abs_SHAP"])


def make_confusion_fig(cm):
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
        x=["Tidak Churn", "Churn"],
        y=["Tidak Churn", "Churn"],
        aspect="auto",
    )
    fig.update_layout(height=420)
    return fig

# =====================================================
# SIDEBAR: AUTO DETECT + UPLOAD OPSIONAL
# =====================================================
st.sidebar.markdown("# 💳 Churn Prediction")
st.sidebar.caption("CatBoost + Bayesian Optimization + SHAP")
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
st.sidebar.markdown("### File Sistem")
uploaded_dataset = st.sidebar.file_uploader("Upload dataset CSV jika belum ada", type=["csv"])
uploaded_model = st.sidebar.file_uploader("Upload model jika belum ada", type=["pkl", "joblib", "cbm"])

# load dataset
found_data_path = find_existing_file(DATA_CANDIDATES)
df = None
if uploaded_dataset is not None:
    df = pd.read_csv(uploaded_dataset)
elif found_data_path is not None:
    df = load_dataset_from_path(str(found_data_path))

# load model
found_model_path = find_existing_file(MODEL_CANDIDATES)
model = None
model_load_error = None
try:
    if uploaded_model is not None:
        model = load_model_from_upload(uploaded_model)
    elif found_model_path is not None:
        model = load_model_from_path(str(found_model_path))
except Exception as e:
    model_load_error = str(e)

best_params = load_pickle_optional(BEST_PARAMS_CANDIDATES)
metrics_data = load_pickle_optional(METRICS_CANDIDATES)

with st.sidebar.expander("Status file", expanded=False):
    if df is not None:
        st.success("Dataset terbaca")
        if found_data_path and uploaded_dataset is None:
            st.caption(str(found_data_path))
    else:
        st.warning("Dataset belum terbaca")
    if model is not None:
        st.success("Model terbaca")
        if found_model_path and uploaded_model is None:
            st.caption(str(found_model_path))
    else:
        st.warning("Model belum terbaca")
        if model_load_error:
            st.error(model_load_error)

# =====================================================
# PESAN STATUS YANG LEBIH JELAS
# =====================================================
if df is None:
    st.info(
        "Dataset belum ditemukan. Taruh file CSV di `data/credit_card_churn.csv`, `data/BankChurners.csv`, "
        "atau upload lewat sidebar."
    )
if model is None:
    st.info(
        "Model belum ditemukan. Taruh model di `model/catboost_bayes_model.pkl` atau upload file model lewat sidebar."
    )

# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":
    title(
        "Prediksi Churn Nasabah Kartu Kredit",
        "Sistem prediksi churn menggunakan CatBoost hasil Optimasi Bayesian dengan interpretasi SHAP.",
    )

    total_data = len(df) if df is not None else 0
    churn_rate = "-"
    auc_value = "-"
    if df is not None and TARGET_COLUMN in df.columns:
        churn_rate = f"{normalize_target(df[TARGET_COLUMN]).mean() * 100:.2f}%"
    if metrics_data and isinstance(metrics_data, dict) and "roc_auc" in metrics_data:
        auc_value = f"{metrics_data['roc_auc']:.4f}"

    c1, c2, c3, c4 = st.columns(4)
    for col, value, label in [
        (c1, total_data, "Jumlah Data"),
        (c2, len(FEATURE_COLUMNS), "Jumlah Fitur"),
        (c3, churn_rate, "Persentase Churn"),
        (c4, auc_value, "ROC-AUC"),
    ]:
        col.markdown(
            f"<div class='metric-card'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        html_card(
            """
            <h3>Tujuan Sistem</h3>
            <p>Sistem ini membantu memprediksi kemungkinan nasabah kartu kredit melakukan churn.
            Output sistem terdiri dari status prediksi, probabilitas churn, tingkat risiko,
            faktor penyebab berdasarkan SHAP, dan rekomendasi retensi.</p>
            """
        )
        html_card(
            """
            <h3>Alur Sistem</h3>
            <p>Dataset → Preprocessing → Bayesian Optimization → CatBoost Final → Evaluasi → SHAP → Prediksi → Rekomendasi Retensi.</p>
            """
        )
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Komponen": ["Algoritma", "Optimasi", "Interpretasi", "Input", "Output"],
                    "Keterangan": ["CatBoost Classifier", "Bayesian Optimization", "SHAP", "Manual & CSV", "Churn / Tidak Churn"],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# =====================================================
# DATASET & EDA
# =====================================================
elif menu == "Dataset & EDA":
    title("Dataset & EDA", "Informasi dataset, distribusi target, dan visualisasi fitur.")
    if df is None:
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Informasi Dataset", "Target", "Analisis Fitur"])
    with tab1:
        section("Preview Dataset")
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", df.shape[0])
        c2.metric("Jumlah Kolom", df.shape[1])
        c3.metric("Missing Value", int(df.isna().sum().sum()))

        section("Deskripsi Fitur")
        st.dataframe(
            pd.DataFrame([{"Fitur": k, "Keterangan": v} for k, v in FEATURE_DESCRIPTIONS.items()]),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        if TARGET_COLUMN not in df.columns:
            st.warning(f"Kolom target `{TARGET_COLUMN}` tidak ada pada dataset.")
        else:
            counts = df[TARGET_COLUMN].value_counts().reset_index()
            counts.columns = ["Status", "Jumlah"]
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(counts, x="Status", y="Jumlah", text="Jumlah", title="Distribusi Target"), use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(counts, names="Status", values="Jumlah", hole=.45, title="Proporsi Target"), use_container_width=True)

    with tab3:
        available_num = [c for c in NUMERIC_COLUMNS if c in df.columns]
        available_cat = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        if available_num:
            chosen_num = st.selectbox("Pilih fitur numerik", available_num)
            st.plotly_chart(
                px.histogram(df, x=chosen_num, color=TARGET_COLUMN if TARGET_COLUMN in df.columns else None, marginal="box"),
                use_container_width=True,
            )
            section("Korelasi Numerik")
            st.plotly_chart(px.imshow(df[available_num].corr(numeric_only=True), aspect="auto"), use_container_width=True)
        if available_cat and TARGET_COLUMN in df.columns:
            chosen_cat = st.selectbox("Pilih fitur kategorikal", available_cat)
            grouped = df.groupby([chosen_cat, TARGET_COLUMN]).size().reset_index(name="Jumlah")
            st.plotly_chart(px.bar(grouped, x=chosen_cat, y="Jumlah", color=TARGET_COLUMN, barmode="group"), use_container_width=True)

# =====================================================
# MODEL FINAL
# =====================================================
elif menu == "Model Final":
    title("Model Final", "Model yang digunakan adalah CatBoost dengan hyperparameter terbaik hasil Bayesian Optimization.")
    html_card(
        """
        <h3>Deskripsi Model</h3>
        <p>Web ini menggunakan model final hasil training dan Optimasi Bayesian. Proses training tidak dijalankan ulang di web agar sistem lebih ringan dan stabil.</p>
        """
    )
    section("Status Model")
    if model is not None:
        st.success("Model berhasil dimuat dan siap digunakan untuk prediksi.")
    else:
        st.warning("Model belum dimuat. Upload lewat sidebar atau simpan ke folder `model/`.")

    section("Hyperparameter Terbaik")
    if isinstance(best_params, dict):
        st.dataframe(pd.DataFrame([{"Hyperparameter": k, "Nilai": v} for k, v in best_params.items()]), use_container_width=True, hide_index=True)
    else:
        st.info("File `best_params.pkl` belum tersedia. Bagian ini opsional.")
        st.dataframe(pd.DataFrame({"Hyperparameter": ["iterations", "depth", "learning_rate", "l2_leaf_reg"], "Nilai": ["...", "...", "...", "..."]}), use_container_width=True, hide_index=True)

    section("Fitur yang Digunakan")
    st.dataframe(pd.DataFrame({"Fitur": FEATURE_COLUMNS}), use_container_width=True, hide_index=True)

# =====================================================
# EVALUASI
# =====================================================
elif menu == "Evaluasi Model":
    title("Evaluasi Model", "Evaluasi performa model final pada dataset yang tersedia.")
    if df is None or model is None:
        st.stop()
    if TARGET_COLUMN not in df.columns:
        st.error(f"Kolom target `{TARGET_COLUMN}` tidak ditemukan.")
        st.stop()

    try:
        X = prepare_features(df)
        y = normalize_target(df[TARGET_COLUMN])
        p = get_prediction_proba(model, X)
        pred = (p >= 0.5).astype(int)

        acc = accuracy_score(y, pred)
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        auc = roc_auc_score(y, p)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")
        c5.metric("ROC-AUC", f"{auc:.4f}")

        left, right = st.columns(2)
        with left:
            section("Confusion Matrix")
            st.plotly_chart(make_confusion_fig(confusion_matrix(y, pred)), use_container_width=True)
        with right:
            section("ROC Curve")
            fpr, tpr, _ = roc_curve(y, p)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc:.4f}"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=420)
            st.plotly_chart(fig, use_container_width=True)

        section("Classification Report")
        report = classification_report(y, pred, target_names=["Tidak Churn", "Churn"], output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
    except Exception as e:
        st.error(f"Evaluasi gagal: {e}")

# =====================================================
# SHAP
# =====================================================
elif menu == "Interpretasi SHAP":
    title("Interpretasi SHAP", "Menjelaskan faktor yang memengaruhi hasil prediksi model.")
    if df is None or model is None:
        st.stop()

    X = prepare_features(df)
    max_sample = min(1000, len(X))
    sample_size = st.slider("Jumlah sampel SHAP", 50, max_sample, min(300, max_sample), step=50)
    X_sample = X.sample(sample_size, random_state=42)

    tab1, tab2 = st.tabs(["Global", "Lokal"])
    with tab1:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            importance = pd.DataFrame({
                "Fitur": X_sample.columns,
                "Mean Absolute SHAP": np.abs(shap_values).mean(axis=0),
            }).sort_values("Mean Absolute SHAP", ascending=False)

            st.plotly_chart(px.bar(importance.head(15), x="Mean Absolute SHAP", y="Fitur", orientation="h", title="Top 15 Fitur SHAP"), use_container_width=True)
            st.dataframe(importance, use_container_width=True, hide_index=True)

            section("SHAP Summary Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, show=False)
            st.pyplot(fig, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            st.error(f"SHAP global gagal dihitung: {e}")

    with tab2:
        idx = st.number_input("Index data nasabah", min_value=0, max_value=len(X)-1, value=0)
        row = X.iloc[[idx]]
        try:
            p = float(get_prediction_proba(model, row)[0])
            pred = "Churn" if p >= .5 else "Tidak Churn"
            risk = risk_level(p)
            c1, c2, c3 = st.columns(3)
            c1.metric("Prediksi", pred)
            c2.metric("Probabilitas Churn", f"{p*100:.2f}%")
            c3.markdown(risk_badge(risk), unsafe_allow_html=True)
            st.dataframe(row, use_container_width=True)
            factors = top_shap_factors(model, row, 8)
            if not factors.empty:
                st.plotly_chart(px.bar(factors.sort_values("Abs_SHAP"), x="Nilai_SHAP", y="Fitur", orientation="h"), use_container_width=True)
                st.dataframe(factors, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"SHAP lokal gagal dihitung: {e}")

# =====================================================
# PREDIKSI
# =====================================================
elif menu == "Prediksi Churn":
    title("Prediksi Churn", "Prediksi melalui input manual atau upload CSV.")
    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])
    with tab1:
        with st.form("manual_form"):
            section("Data Demografis")
            c1, c2, c3 = st.columns(3)
            with c1:
                Customer_Age = st.number_input("Usia", 18, 100, 45)
                Gender = st.selectbox("Gender", ["M", "F"])
            with c2:
                Dependent_count = st.number_input("Jumlah Tanggungan", 0, 10, 2)
                Education_Level = st.selectbox("Pendidikan", ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"])
            with c3:
                Marital_Status = st.selectbox("Status Pernikahan", ["Unknown", "Single", "Married", "Divorced"])
                Income_Category = st.selectbox("Pendapatan", ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"])

            section("Data Kartu dan Transaksi")
            c1, c2, c3 = st.columns(3)
            with c1:
                Card_Category = st.selectbox("Jenis Kartu", ["Blue", "Silver", "Gold", "Platinum"])
                Months_on_book = st.number_input("Months on Book", 1, 80, 36)
                Total_Relationship_Count = st.number_input("Total Relationship", 1, 10, 4)
            with c2:
                Months_Inactive_12_mon = st.number_input("Months Inactive", 0, 12, 2)
                Contacts_Count_12_mon = st.number_input("Contacts Count", 0, 12, 2)
                Credit_Limit = st.number_input("Credit Limit", 0.0, 50000.0, 5000.0, step=100.0)
                Total_Revolving_Bal = st.number_input("Total Revolving Bal", 0.0, 5000.0, 1000.0, step=50.0)
            with c3:
                Avg_Open_To_Buy = st.number_input("Avg Open To Buy", 0.0, 50000.0, 4000.0, step=100.0)
                Total_Amt_Chng_Q4_Q1 = st.number_input("Total Amt Chng Q4/Q1", 0.0, 5.0, .75, step=.01)
                Total_Trans_Amt = st.number_input("Total Trans Amt", 0.0, 25000.0, 4000.0, step=100.0)
                Total_Trans_Ct = st.number_input("Total Trans Ct", 0, 200, 60)
                Total_Ct_Chng_Q4_Q1 = st.number_input("Total Ct Chng Q4/Q1", 0.0, 5.0, .70, step=.01)
                Avg_Utilization_Ratio = st.number_input("Avg Utilization Ratio", 0.0, 1.0, .30, step=.01)

            submitted = st.form_submit_button("Prediksi", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([{col: locals()[col] for col in FEATURE_COLUMNS}])
            X_input = prepare_features(input_df)
            p = float(get_prediction_proba(model, X_input)[0])
            pred = "Churn" if p >= .5 else "Tidak Churn"
            risk = risk_level(p)
            factors = top_shap_factors(model, X_input, 5)
            factor_names = factors["Fitur"].tolist() if not factors.empty else []
            box = "pred-churn" if pred == "Churn" else "pred-safe"

            st.markdown(
                f"<div class='{box}'><h2>Hasil Prediksi: {pred}</h2><h3>Probabilitas Churn: {p*100:.2f}%</h3>{risk_badge(risk)}</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                section("Faktor Utama")
                if not factors.empty:
                    st.dataframe(factors, use_container_width=True, hide_index=True)
                    st.plotly_chart(px.bar(factors, x="Abs_SHAP", y="Fitur", orientation="h"), use_container_width=True)
                else:
                    st.info("Faktor SHAP belum bisa dihitung untuk model ini.")
            with right:
                section("Rekomendasi")
                for r in recommendations(risk, factor_names):
                    st.write(f"- {r}")

    with tab2:
        st.download_button("Download Template CSV", data=template_csv(), file_name="template_prediksi_churn.csv", mime="text/csv", use_container_width=True)
        csv_file = st.file_uploader("Upload file CSV", type=["csv"], key="batch_upload")
        if csv_file is not None:
            batch_df = pd.read_csv(csv_file)
            st.dataframe(batch_df.head(20), use_container_width=True)
            missing = validate_columns(batch_df)
            if missing:
                st.error("Kolom yang belum ada: " + ", ".join(missing))
            elif st.button("Proses Prediksi Batch", use_container_width=True):
                X_batch = prepare_features(batch_df)
                probs = get_prediction_proba(model, X_batch)
                preds = np.where(probs >= .5, "Churn", "Tidak Churn")
                risks = [risk_level(float(x)) for x in probs]
                result = batch_df.copy()
                result["Prediksi"] = preds
                result["Probabilitas_Churn"] = probs
                result["Probabilitas_Churn_Persen"] = [f"{x*100:.2f}%" for x in probs]
                result["Risiko"] = risks

                factor_col, rec_col = [], []
                for i in range(len(result)):
                    if i < 50:
                        f = top_shap_factors(model, X_batch.iloc[[i]], 3)
                        names = f["Fitur"].tolist() if not f.empty else []
                    else:
                        names = []
                    factor_col.append(", ".join(names) if names else "-")
                    rec_col.append("; ".join(recommendations(risks[i], names)))
                result["Faktor_Utama"] = factor_col
                result["Rekomendasi"] = rec_col

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Data", len(result))
                c2.metric("Churn", int((result["Prediksi"] == "Churn").sum()))
                c3.metric("Tidak Churn", int((result["Prediksi"] == "Tidak Churn").sum()))
                st.plotly_chart(px.histogram(result, x="Risiko", color="Risiko", title="Distribusi Tingkat Risiko"), use_container_width=True)
                st.dataframe(result, use_container_width=True)
                st.download_button("Download Hasil Prediksi", data=result.to_csv(index=False).encode("utf-8"), file_name="hasil_prediksi_churn.csv", mime="text/csv", use_container_width=True)

# =====================================================
# REKOMENDASI
# =====================================================
elif menu == "Rekomendasi Retensi":
    title("Rekomendasi Retensi", "Strategi retensi berdasarkan risiko churn dan faktor utama SHAP.")
    st.dataframe(
        pd.DataFrame(
            {
                "Risiko": ["Rendah", "Sedang", "Tinggi"],
                "Kondisi": ["Probabilitas < 40%", "Probabilitas 40% sampai 69%", "Probabilitas >= 70%"],
                "Rekomendasi": [
                    "Monitoring berkala dan pertahankan layanan.",
                    "Berikan reminder, promo ringan, atau reward transaksi.",
                    "Hubungi nasabah, identifikasi keluhan, dan berikan penawaran retensi khusus.",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    section("Rekomendasi berdasarkan Faktor")
    st.dataframe(
        pd.DataFrame(
            {
                "Faktor": [
                    "Months_Inactive_12_mon tinggi",
                    "Total_Trans_Ct rendah",
                    "Total_Trans_Amt rendah",
                    "Contacts_Count_12_mon tinggi",
                    "Avg_Utilization_Ratio tidak optimal",
                ],
                "Rekomendasi": [
                    "Program aktivasi kembali dan reminder penggunaan kartu.",
                    "Promo frekuensi transaksi atau reward point.",
                    "Promo merchant, cicilan ringan, atau cashback nominal tertentu.",
                    "Follow-up keluhan dan evaluasi kualitas layanan.",
                    "Tinjau pola limit dan edukasi penggunaan kartu.",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
