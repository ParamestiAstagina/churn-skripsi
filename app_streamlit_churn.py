# ============================================================
# STREAMLIT APP - Prediksi Customer Churn BankChurners
# Berdasarkan notebook: churn_skripsi_fix.ipynb
# Dataset: BankChurners.csv
# ============================================================

import math
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
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Customer Churn Prediction - CatBoost",
    page_icon="🏦",
    layout="wide",
)

sns.set_style("whitegrid")

# ============================================================
# KONSTANTA SESUAI NOTEBOOK
# ============================================================
DEFAULT_DATA_PATH = Path("BankChurners.csv")

COLS_TO_DROP = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

TARGET_COL = "Attrition_Flag"
TARGET_MAPPING = {
    "Existing Customer": 0,
    "Attrited Customer": 1,
}
TARGET_LABEL = {
    0: "Existing Customer / Tidak Churn",
    1: "Attrited Customer / Churn",
}

# Parameter terbaik dari hasil Bayesian Optimization pada notebook
BEST_PARAMS_BO = {
    "iterations": 435,
    "learning_rate": 0.08545901744057517,
    "depth": 5,
    "l2_leaf_reg": 3.3903849118545817,
}
BEST_THRESHOLD_BO = 0.37

# ============================================================
# FUNGSI LOAD & PENGOLAHAN DATA
# ============================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None) -> pd.DataFrame:
    """Load dataset dari upload Streamlit atau file lokal BankChurners.csv."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if DEFAULT_DATA_PATH.exists():
        return pd.read_csv(DEFAULT_DATA_PATH)

    # Fallback untuk environment ChatGPT/sandbox
    sandbox_path = Path("/mnt/data/BankChurners.csv")
    if sandbox_path.exists():
        return pd.read_csv(sandbox_path)

    raise FileNotFoundError(
        "File BankChurners.csv tidak ditemukan. Upload file CSV melalui sidebar."
    )


@st.cache_data(show_spinner=False)
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Pengolahan data sesuai notebook skripsi."""
    df = raw_df.copy()

    # 1. Seleksi atribut: hapus ID dan 2 kolom Naive Bayes bawaan dataset
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    # 2. Transformasi target: Existing Customer -> 0, Attrited Customer -> 1
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAPPING)

    return df


@st.cache_data(show_spinner=False)
def get_feature_columns(df: pd.DataFrame):
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if TARGET_COL in numerical_cols:
        numerical_cols.remove(TARGET_COL)
    if TARGET_COL in categorical_cols:
        categorical_cols.remove(TARGET_COL)

    return numerical_cols, categorical_cols


@st.cache_data(show_spinner=False)
def check_outliers_table(df: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if TARGET_COL in numerical_cols:
        numerical_cols.remove(TARGET_COL)

    outlier_summary = []
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        outlier_summary.append(
            {
                "Atribut": col,
                "Jumlah Outlier": outlier_count,
                "Lower Bound": lower_bound,
                "Upper Bound": upper_bound,
            }
        )

    return pd.DataFrame(outlier_summary)


# ============================================================
# TRAINING MODEL
# ============================================================
@st.cache_resource(show_spinner=True)
def train_models(df: pd.DataFrame):
    """Training CatBoost baseline dan CatBoost + BO sesuai pengolahan notebook."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    # Model 1: CatBoost Baseline
    cb_baseline = CatBoostClassifier(
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
    )
    cb_baseline.fit(
        X_tr,
        y_tr,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    y_pred_base = cb_baseline.predict(X_test).astype(int)
    y_prob_base = cb_baseline.predict_proba(X_test)[:, 1]

    # Model 2: CatBoost + Bayesian Optimization params dari notebook
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )

    cb_bo = CatBoostClassifier(
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=42,
        **BEST_PARAMS_BO,
    )
    cb_bo.fit(
        X_tr_final,
        y_tr_final,
        cat_features=cat_features,
        eval_set=(X_val_final, y_val_final),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    y_prob_bo = cb_bo.predict_proba(X_test)[:, 1]
    y_pred_bo = (y_prob_bo >= BEST_THRESHOLD_BO).astype(int)

    metrics = pd.DataFrame(
        [
            evaluate_model("CatBoost Baseline", y_test, y_pred_base, y_prob_base),
            evaluate_model("CatBoost + Bayesian Optimization", y_test, y_pred_bo, y_prob_bo),
        ]
    )

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cat_features": cat_features,
        "baseline_model": cb_baseline,
        "bo_model": cb_bo,
        "y_pred_base": y_pred_base,
        "y_prob_base": y_prob_base,
        "y_pred_bo": y_pred_bo,
        "y_prob_bo": y_prob_bo,
        "metrics": metrics,
    }


def evaluate_model(model_name, y_true, y_pred, y_prob):
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "F2-Score": fbeta_score(y_true, y_pred, beta=2),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


# ============================================================
# VISUALISASI
# ============================================================
def plot_count_target(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    target_display = df[TARGET_COL].map(TARGET_LABEL)
    sns.countplot(x=target_display, ax=ax)
    ax.set_title("Distribusi Attrition_Flag")
    ax.set_xlabel("Status Nasabah")
    ax.set_ylabel("Jumlah Nasabah")
    ax.tick_params(axis="x", rotation=10)

    for container in ax.containers:
        ax.bar_label(container, fmt="%d")

    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    plt.tight_layout()
    return fig


def plot_metric_comparison(metrics_df):
    metric_order = ["Accuracy", "Precision", "Recall", "F1-Score", "F2-Score", "ROC-AUC"]
    df_melted = metrics_df.melt(
        id_vars="Model",
        value_vars=metric_order,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Perbandingan Kinerja Model")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Metrik")
    ax.set_ylabel("Skor")
    ax.tick_params(axis="x", rotation=20)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    plt.tight_layout()
    return fig


def plot_numeric_distribution(df, selected_col):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x=selected_col, kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribusi {selected_col}")
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Frekuensi")
    plt.tight_layout()
    return fig


def plot_categorical_distribution(df, selected_col):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = df[selected_col].value_counts().index
    sns.countplot(data=df, x=selected_col, order=order, ax=ax)
    ax.set_title(f"Distribusi {selected_col}")
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_numeric_vs_target(df, selected_col):
    fig, ax = plt.subplots(figsize=(7, 4))
    temp = df.copy()
    temp["Status"] = temp[TARGET_COL].map(TARGET_LABEL)
    sns.boxplot(data=temp, x="Status", y=selected_col, ax=ax)
    ax.set_title(f"{selected_col} vs Attrition_Flag")
    ax.set_xlabel("Status Nasabah")
    ax.set_ylabel(selected_col)
    ax.tick_params(axis="x", rotation=10)
    plt.tight_layout()
    return fig


def plot_categorical_vs_target(df, selected_col):
    fig, ax = plt.subplots(figsize=(8, 4))
    temp = df.copy()
    temp["Status"] = temp[TARGET_COL].map(TARGET_LABEL)
    order = temp[selected_col].value_counts().index
    sns.countplot(data=temp, x=selected_col, hue="Status", order=order, ax=ax)
    ax.set_title(f"{selected_col} vs Attrition_Flag")
    ax.set_xlabel(selected_col)
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, numerical_cols):
    corr_matrix = df[numerical_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Heatmap Korelasi Antar Fitur Numerik")
    plt.tight_layout()
    return fig


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload BankChurners.csv", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Aplikasi ini menggunakan pengolahan data dari notebook: seleksi atribut, mapping target, split stratified, CatBoost baseline, dan CatBoost + BO."
)

# ============================================================
# LOAD DATA
# ============================================================
try:
    raw_df = load_data(uploaded_file)
    df = preprocess_data(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

if TARGET_COL not in df.columns:
    st.error(f"Kolom target '{TARGET_COL}' tidak ditemukan pada dataset.")
    st.stop()

numerical_cols, categorical_cols = get_feature_columns(df)

# ============================================================
# HEADER
# ============================================================
st.title("🏦 Customer Churn Prediction dengan CatBoost")
st.write(
    "Dashboard Streamlit ini menampilkan pengolahan data, EDA, training model, evaluasi, "
    "dan prediksi churn nasabah berdasarkan dataset BankChurners."
)

# ============================================================
# RINGKASAN DATA
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
col2.metric("Jumlah Kolom Setelah Seleksi", f"{df.shape[1]:,}")
col3.metric("Fitur Numerik", len(numerical_cols))
col4.metric("Fitur Kategorikal", len(categorical_cols))

# ============================================================
# TABS
# ============================================================
tab_data, tab_eda, tab_model, tab_predict, tab_code = st.tabs(
    [
        "📄 Data & Pengolahan",
        "📊 EDA",
        "🤖 Model & Evaluasi",
        "🔮 Prediksi Nasabah",
        "🧾 Ringkasan Kode",
    ]
)

# ============================================================
# TAB 1: DATA & PENGOLAHAN
# ============================================================
with tab_data:
    st.subheader("Data Awal")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.subheader("Tahapan Pengolahan Data yang Dilakukan")
    st.markdown(
        """
        1. **Load dataset** `BankChurners.csv`.
        2. **Seleksi atribut** dengan menghapus:
           - `CLIENTNUM`
           - 2 kolom `Naive_Bayes_Classifier_...`
        3. **Cek missing value**.
        4. **Cek duplikasi data**.
        5. **Cek nilai `Unknown`** pada kolom kategorikal.
        6. **Cek outlier** menggunakan metode IQR.
        7. **Transformasi target**:
           - `Existing Customer` → `0`
           - `Attrited Customer` → `1`
        8. **Pemisahan fitur dan target**.
        9. **Split data**: 80% training dan 20% testing dengan stratifikasi.
        """
    )

    st.subheader("Data Setelah Seleksi Atribut dan Transformasi Target")
    st.dataframe(df.head(20), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Missing Value per Kolom**")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Kolom", "Jumlah Missing"]
        st.dataframe(missing_df, use_container_width=True)

    with c2:
        st.write("**Jumlah Data Duplikat**")
        st.metric("Duplikasi", int(df.duplicated().sum()))

    st.subheader("Nilai `Unknown` pada Kolom Kategorikal")
    unknown_rows = []
    for col in df.select_dtypes(include=["object"]).columns:
        unknown_rows.append({"Kolom": col, "Jumlah Unknown": int((df[col] == "Unknown").sum())})
    st.dataframe(pd.DataFrame(unknown_rows), use_container_width=True)

    st.subheader("Outlier dengan Metode IQR")
    outlier_df = check_outliers_table(df)
    st.dataframe(outlier_df, use_container_width=True)

# ============================================================
# TAB 2: EDA
# ============================================================
with tab_eda:
    st.subheader("Distribusi Target")
    st.pyplot(plot_count_target(df))

    st.subheader("Distribusi Fitur")
    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        selected_num = st.selectbox("Pilih fitur numerik", numerical_cols)
        st.pyplot(plot_numeric_distribution(df, selected_num))
        st.pyplot(plot_numeric_vs_target(df, selected_num))

    with eda_col2:
        selected_cat = st.selectbox("Pilih fitur kategorikal", categorical_cols)
        st.pyplot(plot_categorical_distribution(df, selected_cat))
        st.pyplot(plot_categorical_vs_target(df, selected_cat))

    st.subheader("Korelasi Antar Fitur Numerik")
    st.pyplot(plot_correlation_heatmap(df, numerical_cols))

# ============================================================
# TAB 3: MODEL & EVALUASI
# ============================================================
with tab_model:
    st.subheader("Training dan Evaluasi Model")
    st.info(
        "Model akan dilatih menggunakan CatBoost. Proses training memakai cache Streamlit, "
        "jadi hanya berjalan ulang jika data berubah."
    )

    with st.spinner("Melatih model CatBoost..."):
        model_artifacts = train_models(df)

    metrics_df = model_artifacts["metrics"]
    st.subheader("Perbandingan Metrik")
    st.dataframe(metrics_df.style.format({col: "{:.4f}" for col in metrics_df.columns if col != "Model"}), use_container_width=True)
    st.pyplot(plot_metric_comparison(metrics_df))

    st.subheader("Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.pyplot(
            plot_confusion_matrix(
                model_artifacts["y_test"],
                model_artifacts["y_pred_base"],
                "Confusion Matrix - CatBoost Baseline",
            )
        )
    with cm_col2:
        st.pyplot(
            plot_confusion_matrix(
                model_artifacts["y_test"],
                model_artifacts["y_pred_bo"],
                "Confusion Matrix - CatBoost + BO",
            )
        )

    st.subheader("Classification Report")
    report_choice = st.radio(
        "Pilih model",
        ["CatBoost Baseline", "CatBoost + Bayesian Optimization"],
        horizontal=True,
    )

    if report_choice == "CatBoost Baseline":
        report = classification_report(
            model_artifacts["y_test"],
            model_artifacts["y_pred_base"],
            target_names=["Stay", "Churn"],
            output_dict=True,
        )
    else:
        report = classification_report(
            model_artifacts["y_test"],
            model_artifacts["y_pred_bo"],
            target_names=["Stay", "Churn"],
            output_dict=True,
        )
    st.dataframe(pd.DataFrame(report).T, use_container_width=True)

    st.subheader("Parameter Model Optimasi")
    st.json({"best_params_bo": BEST_PARAMS_BO, "best_threshold": BEST_THRESHOLD_BO})

# ============================================================
# TAB 4: PREDIKSI NASABAH
# ============================================================
with tab_predict:
    st.subheader("Prediksi Churn untuk Satu Nasabah")

    with st.spinner("Menyiapkan model prediksi..."):
        model_artifacts = train_models(df)

    model_choice = st.selectbox(
        "Pilih model untuk prediksi",
        ["CatBoost Baseline", "CatBoost + Bayesian Optimization"],
    )

    input_data = {}
    st.write("Masukkan nilai fitur nasabah:")

    form_cols = st.columns(2)
    for idx, col in enumerate(model_artifacts["X"].columns):
        target_container = form_cols[idx % 2]
        with target_container:
            if col in categorical_cols:
                options = sorted(df[col].dropna().unique().tolist())
                default_idx = 0 if options else None
                input_data[col] = st.selectbox(col, options, index=default_idx)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                median_val = float(df[col].median())
                input_data[col] = st.number_input(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=median_val,
                    step=1.0 if pd.api.types.is_integer_dtype(df[col]) else 0.01,
                )

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi Churn", type="primary"):
        if model_choice == "CatBoost Baseline":
            model = model_artifacts["baseline_model"]
            prob_churn = model.predict_proba(input_df)[:, 1][0]
            pred = int(model.predict(input_df)[0])
        else:
            model = model_artifacts["bo_model"]
            prob_churn = model.predict_proba(input_df)[:, 1][0]
            pred = int(prob_churn >= BEST_THRESHOLD_BO)

        st.subheader("Hasil Prediksi")
        st.metric("Probabilitas Churn", f"{prob_churn:.2%}")

        if pred == 1:
            st.error("Prediksi: Nasabah berpotensi CHURN / Attrited Customer")
        else:
            st.success("Prediksi: Nasabah cenderung TIDAK CHURN / Existing Customer")

        st.write("**Data input nasabah:**")
        st.dataframe(input_df, use_container_width=True)

# ============================================================
# TAB 5: RINGKASAN KODE
# ============================================================
with tab_code:
    st.subheader("Kode Inti Pengolahan Data")
    st.code(
        """
# Seleksi atribut
cols_to_drop = [
    'CLIENTNUM',
    'Naive_Bayes_Classifier_..._1',
    'Naive_Bayes_Classifier_..._2'
]
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Transformasi target
df['Attrition_Flag'] = df['Attrition_Flag'].map({
    'Existing Customer': 0,
    'Attrited Customer': 1
})

# Fitur dan target
X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost Baseline
cb_baseline = CatBoostClassifier(
    auto_class_weights='Balanced',
    random_seed=42,
    verbose=0
)

# CatBoost + Bayesian Optimization
best_params = {
    'iterations': 435,
    'learning_rate': 0.08545901744057517,
    'depth': 5,
    'l2_leaf_reg': 3.3903849118545817
}
best_threshold = 0.37
        """,
        language="python",
    )

    st.subheader("Cara Menjalankan")
    st.code(
        """
pip install streamlit pandas numpy matplotlib seaborn scikit-learn catboost
streamlit run app_streamlit_churn.py
        """,
        language="bash",
    )
