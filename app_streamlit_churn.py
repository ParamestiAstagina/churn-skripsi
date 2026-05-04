# ============================================================
# STREAMLIT APP INTERAKTIF - CUSTOMER CHURN BANKCHURNERS
# Fitur:
# 1. Bisa memakai dataset default BankChurners.csv
# 2. Bisa upload file CSV sendiri
# 3. Tampilan dashboard lebih rapi
# 4. Pengolahan data sesuai notebook churn_skripsi_fix.ipynb
# 5. Training CatBoost
# 6. Input manual untuk prediksi 1 nasabah
# 7. Upload file CSV untuk prediksi batch
# ============================================================

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Bank Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_style("whitegrid")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .main {
        background: #f7f9fc;
    }
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
        color: white;
        padding: 28px 32px;
        border-radius: 24px;
        margin-bottom: 24px;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.22);
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.92;
        line-height: 1.55;
        max-width: 980px;
    }
    .section-card {
        background: white;
        padding: 18px 20px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 7px 22px rgba(15, 23, 42, 0.05);
        margin-bottom: 16px;
    }
    .small-muted {
        color: #64748b;
        font-size: 13px;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 16px 18px;
        border-radius: 18px;
        box-shadow: 0 7px 20px rgba(15, 23, 42, 0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 800;
    }
    .prediction-churn {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        border-radius: 18px;
        padding: 18px 20px;
        font-size: 18px;
        font-weight: 700;
        text-align: center;
    }
    .prediction-safe {
        background: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
        border-radius: 18px;
        padding: 18px 20px;
        font-size: 18px;
        font-weight: 700;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 14px 14px 0 0;
        padding: 10px 16px;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONSTANTS
# ============================================================
DEFAULT_DATA_PATHS = [Path("BankChurners.csv"), Path("/mnt/data/BankChurners.csv")]

TARGET_COL = "Attrition_Flag"
TARGET_MAPPING = {"Existing Customer": 0, "Attrited Customer": 1}
TARGET_REVERSE = {0: "Existing Customer", 1: "Attrited Customer"}

COLS_TO_DROP = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

BEST_PARAMS_BO = {
    "iterations": 435,
    "learning_rate": 0.08545901744057517,
    "depth": 5,
    "l2_leaf_reg": 3.3903849118545817,
}
BEST_THRESHOLD = 0.37

MODEL_LABELS = {
    "baseline": "CatBoost Baseline",
    "optimized": "CatBoost Optimized + Threshold 0.37",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data(show_spinner=False)
def read_default_dataset() -> pd.DataFrame:
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError("Dataset default BankChurners.csv tidak ditemukan.")


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def preprocess_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pengolahan data sesuai notebook: drop kolom tidak dipakai dan mapping target."""
    df = raw_df.copy()
    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors="ignore")

    if TARGET_COL in df.columns:
        if df[TARGET_COL].dtype == "object":
            df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df


@st.cache_data(show_spinner=False)
def get_columns(df: pd.DataFrame):
    feature_df = df.drop(columns=[TARGET_COL], errors="ignore")
    num_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num_cols, cat_cols


@st.cache_data(show_spinner=False)
def data_quality_summary(raw_df: pd.DataFrame, processed_df: pd.DataFrame) -> pd.DataFrame:
    total_unknown = 0
    for col in raw_df.select_dtypes(include=["object"]).columns:
        total_unknown += int((raw_df[col].astype(str).str.lower() == "unknown").sum())

    return pd.DataFrame(
        [
            {"Pemeriksaan": "Jumlah baris data awal", "Nilai": f"{raw_df.shape[0]:,}"},
            {"Pemeriksaan": "Jumlah kolom data awal", "Nilai": f"{raw_df.shape[1]:,}"},
            {"Pemeriksaan": "Jumlah kolom setelah seleksi", "Nilai": f"{processed_df.shape[1]:,}"},
            {"Pemeriksaan": "Missing value", "Nilai": f"{int(processed_df.isna().sum().sum()):,}"},
            {"Pemeriksaan": "Data duplikat", "Nilai": f"{int(processed_df.duplicated().sum()):,}"},
            {"Pemeriksaan": "Nilai Unknown", "Nilai": f"{total_unknown:,}"},
        ]
    )


@st.cache_data(show_spinner=False)
def outlier_summary(df: pd.DataFrame, numerical_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        rows.append(
            {
                "Fitur": col,
                "Jumlah Outlier": count,
                "Persentase": round((count / len(df)) * 100, 2),
                "Lower Bound": round(lower, 3),
                "Upper Bound": round(upper, 3),
            }
        )
    return pd.DataFrame(rows).sort_values("Jumlah Outlier", ascending=False)


def validate_training_data(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        return False, f"Kolom target `{TARGET_COL}` tidak ditemukan. Dataset untuk training harus memiliki kolom target."
    if df[TARGET_COL].nunique() != 2:
        return False, f"Kolom target `{TARGET_COL}` harus memiliki 2 kelas."
    return True, "OK"


@st.cache_resource(show_spinner=False)
def train_catboost_models(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    _, cat_cols = get_columns(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    baseline = CatBoostClassifier(
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
    )
    baseline.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    optimized = CatBoostClassifier(
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        **BEST_PARAMS_BO,
    )
    optimized.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    prob_base = baseline.predict_proba(X_test)[:, 1]
    pred_base = baseline.predict(X_test).astype(int)

    prob_opt = optimized.predict_proba(X_test)[:, 1]
    pred_opt = (prob_opt >= BEST_THRESHOLD).astype(int)

    metrics_df = pd.DataFrame(
        [
            evaluate_model(MODEL_LABELS["baseline"], y_test, pred_base, prob_base),
            evaluate_model(MODEL_LABELS["optimized"], y_test, pred_opt, prob_opt),
        ]
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "cat_cols": cat_cols,
        "baseline": baseline,
        "optimized": optimized,
        "pred_base": pred_base,
        "prob_base": prob_base,
        "pred_opt": pred_opt,
        "prob_opt": prob_opt,
        "metrics": metrics_df,
    }


def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "F2-Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def prepare_prediction_df(input_df: pd.DataFrame, training_columns: list[str]) -> pd.DataFrame:
    df = preprocess_dataset(input_df)
    df = df.drop(columns=[TARGET_COL], errors="ignore")

    missing_cols = [col for col in training_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in training_columns]

    if missing_cols:
        raise ValueError("Kolom berikut belum ada pada file prediksi: " + ", ".join(missing_cols))
    if extra_cols:
        df = df.drop(columns=extra_cols)

    return df[training_columns]


def predict_with_model(model, input_df: pd.DataFrame, threshold: float | None = None):
    prob = model.predict_proba(input_df)[:, 1]
    if threshold is None:
        pred = model.predict(input_df).astype(int)
    else:
        pred = (prob >= threshold).astype(int)
    return pred, prob

# ============================================================
# PLOT FUNCTIONS
# ============================================================
def plot_target_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    temp = df.copy()
    temp["Status"] = temp[TARGET_COL].map(TARGET_REVERSE)
    order = ["Existing Customer", "Attrited Customer"]
    sns.countplot(data=temp, x="Status", order=order, ax=ax)
    ax.set_title("Distribusi Target Attrition_Flag", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Jumlah Nasabah")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    plt.tight_layout()
    return fig


def plot_numeric(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df[col], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribusi {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Frekuensi")
    plt.tight_layout()
    return fig


def plot_categorical(df: pd.DataFrame, col: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, ax=ax)
    ax.set_title(f"Distribusi {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Jumlah")
    ax.set_ylabel(col)
    plt.tight_layout()
    return fig


def plot_feature_vs_target(df: pd.DataFrame, col: str, is_numeric: bool):
    fig, ax = plt.subplots(figsize=(7, 4))
    temp = df.copy()
    temp["Status"] = temp[TARGET_COL].map(TARGET_REVERSE)
    if is_numeric:
        sns.boxplot(data=temp, x="Status", y=col, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(col)
    else:
        order = temp[col].value_counts().index
        sns.countplot(data=temp, y=col, hue="Status", order=order, ax=ax)
        ax.set_xlabel("Jumlah")
        ax.set_ylabel(col)
    ax.set_title(f"{col} berdasarkan Status Churn", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Tidak Churn", "Churn"],
        yticklabels=["Tidak Churn", "Churn"],
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    plt.tight_layout()
    return fig


def plot_metric_bar(metrics_df: pd.DataFrame):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "ROC-AUC"]
    melted = metrics_df.melt(id_vars="Model", value_vars=metric_cols, var_name="Metrik", value_name="Skor")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.barplot(data=melted, x="Metrik", y="Skor", hue="Model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Perbandingan Metrik Model", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Skor")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, num_cols: list[str]):
    fig, ax = plt.subplots(figsize=(11, 7))
    corr = df[num_cols + [TARGET_COL]].corr()
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🏦 Churn Dashboard")
    st.caption("Upload dataset, latih model, dan lakukan prediksi churn nasabah.")
    st.markdown("---")

    data_source = st.radio(
        "Sumber Dataset Training",
        ["Gunakan dataset default", "Upload CSV training"],
        help="Dataset training harus memiliki kolom Attrition_Flag.",
    )

    uploaded_train = None
    if data_source == "Upload CSV training":
        uploaded_train = st.file_uploader("Upload CSV training", type=["csv"], key="train_uploader")

    st.markdown("---")
    model_mode = st.selectbox(
        "Model untuk Prediksi",
        list(MODEL_LABELS.keys()),
        format_func=lambda key: MODEL_LABELS[key],
    )

    st.caption(
        "CatBoost Optimized memakai parameter Bayesian Optimization dan threshold 0.37 sesuai notebook."
    )

# ============================================================
# LOAD DATA
# ============================================================
try:
    if data_source == "Upload CSV training":
        if uploaded_train is None:
            raw_df = read_default_dataset()
            st.sidebar.info("Belum ada file di-upload. Sementara memakai dataset default.")
        else:
            raw_df = read_uploaded_csv(uploaded_train)
    else:
        raw_df = read_default_dataset()
except Exception as exc:
    st.error(f"Gagal membaca dataset: {exc}")
    st.stop()

processed_df = preprocess_dataset(raw_df)
can_train, validation_message = validate_training_data(processed_df)

if not can_train:
    st.error(validation_message)
    st.stop()

numerical_cols, categorical_cols = get_columns(processed_df)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Customer Churn Analytics</div>
        <div class="hero-subtitle">
            Dashboard prediksi churn nasabah kartu kredit menggunakan CatBoost. Aplikasi ini mengikuti alur pengolahan data dari notebook:
            seleksi atribut, mapping target, split stratified, training model, evaluasi, input manual, dan prediksi batch dari file upload.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

existing_count = int((processed_df[TARGET_COL] == 0).sum())
churn_count = int((processed_df[TARGET_COL] == 1).sum())
churn_rate = churn_count / len(processed_df) if len(processed_df) else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Nasabah", f"{len(processed_df):,}")
m2.metric("Jumlah Fitur", f"{processed_df.shape[1] - 1:,}")
m3.metric("Nasabah Churn", f"{churn_count:,}", f"{churn_rate:.2%}")
m4.metric("Nasabah Tidak Churn", f"{existing_count:,}")

# ============================================================
# TRAIN MODEL ONCE
# ============================================================
with st.spinner("Menyiapkan model CatBoost..."):
    artifacts = train_catboost_models(processed_df)

selected_model = artifacts[model_mode]
selected_threshold = BEST_THRESHOLD if model_mode == "optimized" else None

# ============================================================
# TABS
# ============================================================
tab_overview, tab_processing, tab_eda, tab_model, tab_manual, tab_batch = st.tabs(
    [
        "🏠 Overview",
        "🧹 Pengolahan Data",
        "📊 EDA",
        "🤖 Model",
        "✍️ Input Manual",
        "📁 Upload Prediksi",
    ]
)

# ============================================================
# TAB OVERVIEW
# ============================================================
with tab_overview:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Preview Dataset")
        st.dataframe(processed_df.head(15), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Distribusi Churn")
        st.pyplot(plot_target_distribution(processed_df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Dataset")
    st.dataframe(data_quality_summary(raw_df, processed_df), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB PROCESSING
# ============================================================
with tab_processing:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Tahapan Pengolahan Data")
    st.markdown(
        """
        1. Membaca dataset `BankChurners.csv` atau file CSV yang di-upload pengguna.  
        2. Menghapus kolom identitas `CLIENTNUM`.  
        3. Menghapus dua kolom `Naive_Bayes_Classifier_...` bawaan dataset.  
        4. Mengubah target `Attrition_Flag`: `Existing Customer` menjadi `0`, dan `Attrited Customer` menjadi `1`.  
        5. Memisahkan fitur dan target.  
        6. Melakukan split data 80:20 dengan `stratify=y`.  
        7. Melatih model CatBoost baseline dan CatBoost hasil optimasi.  
        8. Menggunakan threshold `0.37` untuk model optimized.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Missing Value")
        missing = processed_df.isna().sum().reset_index()
        missing.columns = ["Kolom", "Jumlah Missing"]
        st.dataframe(missing, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Nilai Unknown")
        unknown_rows = []
        for col in processed_df.select_dtypes(include=["object"]).columns:
            unknown_rows.append({"Kolom": col, "Jumlah Unknown": int((processed_df[col] == "Unknown").sum())})
        if unknown_rows:
            st.dataframe(pd.DataFrame(unknown_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada kolom kategorikal bertipe object setelah pengolahan.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Outlier Metode IQR")
    st.dataframe(outlier_summary(processed_df, numerical_cols), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB EDA
# ============================================================
with tab_eda:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Eksplorasi Fitur")
    eda_left, eda_right = st.columns(2)

    with eda_left:
        selected_num = st.selectbox("Pilih fitur numerik", numerical_cols, key="eda_num")
        st.pyplot(plot_numeric(processed_df, selected_num), use_container_width=True)
        st.pyplot(plot_feature_vs_target(processed_df, selected_num, True), use_container_width=True)

    with eda_right:
        if categorical_cols:
            selected_cat = st.selectbox("Pilih fitur kategorikal", categorical_cols, key="eda_cat")
            st.pyplot(plot_categorical(processed_df, selected_cat), use_container_width=True)
            st.pyplot(plot_feature_vs_target(processed_df, selected_cat, False), use_container_width=True)
        else:
            st.info("Tidak ada fitur kategorikal pada dataset ini.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Korelasi")
    if len(numerical_cols) > 1:
        st.pyplot(plot_correlation(processed_df, numerical_cols), use_container_width=True)
    else:
        st.info("Korelasi membutuhkan minimal dua fitur numerik.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB MODEL
# ============================================================
with tab_model:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Metrik Evaluasi")
    metrics_df = artifacts["metrics"]
    st.dataframe(
        metrics_df.style.format({col: "{:.4f}" for col in metrics_df.columns if col != "Model"}),
        use_container_width=True,
        hide_index=True,
    )
    st.pyplot(plot_metric_bar(metrics_df), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.pyplot(plot_confusion(artifacts["y_test"], artifacts["pred_base"], "CatBoost Baseline"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cm2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.pyplot(plot_confusion(artifacts["y_test"], artifacts["pred_opt"], "CatBoost Optimized"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Classification Report")
    report_model = st.radio(
        "Pilih report model",
        ["baseline", "optimized"],
        horizontal=True,
        format_func=lambda key: MODEL_LABELS[key],
    )
    preds = artifacts["pred_base"] if report_model == "baseline" else artifacts["pred_opt"]
    report = classification_report(
        artifacts["y_test"],
        preds,
        target_names=["Tidak Churn", "Churn"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB MANUAL PREDICTION
# ============================================================
with tab_manual:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Input Manual Data Nasabah")
    st.caption("Isi nilai fitur nasabah, lalu klik tombol prediksi.")

    input_values = {}
    feature_columns = artifacts["X"].columns.tolist()

    with st.form("manual_prediction_form"):
        col_left, col_right = st.columns(2)
        for index, col in enumerate(feature_columns):
            container = col_left if index % 2 == 0 else col_right
            with container:
                if col in categorical_cols:
                    options = sorted(processed_df[col].dropna().astype(str).unique().tolist())
                    default_value = options[0] if options else "Unknown"
                    input_values[col] = st.selectbox(col, options, index=options.index(default_value) if default_value in options else 0)
                else:
                    series = processed_df[col]
                    min_val = float(series.min())
                    max_val = float(series.max())
                    median_val = float(series.median())
                    if pd.api.types.is_integer_dtype(series):
                        input_values[col] = st.number_input(
                            col,
                            min_value=int(min_val),
                            max_value=int(max_val),
                            value=int(round(median_val)),
                            step=1,
                        )
                    else:
                        input_values[col] = st.number_input(
                            col,
                            min_value=min_val,
                            max_value=max_val,
                            value=median_val,
                            step=0.01,
                        )

        submitted = st.form_submit_button("🔮 Prediksi Churn", type="primary")

    if submitted:
        manual_df = pd.DataFrame([input_values])[feature_columns]
        pred, prob = predict_with_model(selected_model, manual_df, selected_threshold)
        prob_churn = float(prob[0])
        pred_label = int(pred[0])

        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Probabilitas Churn", f"{prob_churn:.2%}")
        with r2:
            st.metric("Model Digunakan", MODEL_LABELS[model_mode])

        if pred_label == 1:
            st.markdown('<div class="prediction-churn">Prediksi: Nasabah berpotensi CHURN / Attrited Customer</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-safe">Prediksi: Nasabah cenderung TIDAK CHURN / Existing Customer</div>', unsafe_allow_html=True)

        st.write("Data input:")
        st.dataframe(manual_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB BATCH PREDICTION
# ============================================================
with tab_batch:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Upload File CSV untuk Prediksi Batch")
    st.caption(
        "File prediksi boleh memiliki kolom target atau tidak. Kolom target akan diabaikan. Struktur fitur harus sama dengan dataset training."
    )

    sample_template = artifacts["X"].head(5).copy()
    st.download_button(
        "⬇️ Download Template CSV Prediksi",
        data=make_download_csv(sample_template),
        file_name="template_prediksi_churn.csv",
        mime="text/csv",
    )

    uploaded_predict = st.file_uploader("Upload CSV prediksi", type=["csv"], key="predict_uploader")

    if uploaded_predict is not None:
        try:
            raw_pred_df = pd.read_csv(uploaded_predict)
            pred_df = prepare_prediction_df(raw_pred_df, artifacts["X"].columns.tolist())
            pred, prob = predict_with_model(selected_model, pred_df, selected_threshold)

            result_df = raw_pred_df.copy()
            result_df["Probabilitas_Churn"] = prob
            result_df["Prediksi_Label"] = pred
            result_df["Prediksi_Status"] = result_df["Prediksi_Label"].map(
                {0: "Existing Customer / Tidak Churn", 1: "Attrited Customer / Churn"}
            )

            st.success("Prediksi batch berhasil dibuat.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Jumlah Data", f"{len(result_df):,}")
            c2.metric("Prediksi Churn", f"{int((pred == 1).sum()):,}")
            c3.metric("Rata-rata Prob. Churn", f"{float(np.mean(prob)):.2%}")

            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                "⬇️ Download Hasil Prediksi",
                data=make_download_csv(result_df),
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(f"Gagal melakukan prediksi batch: {exc}")
    else:
        st.info("Upload file CSV prediksi untuk menampilkan hasil prediksi batch.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.caption("Dibuat untuk analisis churn nasabah BankChurners menggunakan Streamlit dan CatBoost.")
