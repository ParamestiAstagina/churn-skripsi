# =========================================================
# APLIKASI STREAMLIT PREDIKSI CUSTOMER CHURN BANK
# Berdasarkan alur pengolahan data pada notebook churn_skripsi_fix.ipynb
# Model: CatBoostClassifier
# Fitur:
# 1. User dapat upload file CSV sendiri
# 2. Data diproses sesuai notebook
# 3. Model dilatih dari data upload
# 4. User dapat input nilai fitur manual untuk prediksi churn
# 5. User dapat prediksi banyak data dari CSV tanpa kolom target
# =========================================================

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
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

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Prediksi Customer Churn Bank",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Prediksi Customer Churn Bank dengan CatBoost")
st.write(
    "Aplikasi ini menggunakan alur preprocessing dari notebook: seleksi atribut, "
    "transformasi target, pemisahan fitur kategorikal, training CatBoost, evaluasi, "
    "dan prediksi churn berdasarkan input manual atau file CSV."
)

# =========================================================
# KONSTANTA
# =========================================================
TARGET_COL = "Attrition_Flag"
POSITIVE_LABEL_NAME = "Attrited Customer"
NEGATIVE_LABEL_NAME = "Existing Customer"

COLS_TO_DROP = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

DEFAULT_CATBOOST_PARAMS = {
    "iterations": 300,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 10,
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": 0,
}

# =========================================================
# FUNGSI PREPROCESSING
# =========================================================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus spasi berlebih pada nama kolom."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleksi atribut sesuai notebook."""
    df = df.copy()
    existing_drop_cols = [col for col in COLS_TO_DROP if col in df.columns]
    return df.drop(columns=existing_drop_cols)


def transform_target(df: pd.DataFrame) -> pd.DataFrame:
    """Transformasi Attrition_Flag: Existing Customer=0, Attrited Customer=1."""
    df = df.copy()

    if TARGET_COL not in df.columns:
        return df

    if df[TARGET_COL].dtype == "object":
        df[TARGET_COL] = df[TARGET_COL].map({
            NEGATIVE_LABEL_NAME: 0,
            POSITIVE_LABEL_NAME: 1,
            "Existing": 0,
            "Attrited": 1,
            "No": 0,
            "Yes": 1,
            "0": 0,
            "1": 1,
        })

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    return df


def preprocess_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing data train sesuai notebook."""
    df = clean_columns(df)
    df = drop_unused_columns(df)
    df = transform_target(df)

    # Hapus baris target kosong/tidak valid
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])
        df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Hapus duplikasi bila ada
    df = df.drop_duplicates()

    return df


def preprocess_prediction_data(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Preprocessing data prediksi agar kolomnya sama dengan data training."""
    df = clean_columns(df)
    df = drop_unused_columns(df)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # Pastikan semua fitur training tersedia
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom berikut belum ada pada data prediksi: {missing_cols}")

    # Ambil dan urutkan kolom sesuai training
    df = df[feature_columns].copy()
    return df


def get_feature_types(df: pd.DataFrame):
    """Mengambil fitur numerik dan kategorikal."""
    feature_df = df.drop(columns=[TARGET_COL], errors="ignore")
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = feature_df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    return numerical_cols, categorical_cols


def check_dataset(df: pd.DataFrame):
    """Validasi dataset training."""
    if TARGET_COL not in df.columns:
        st.error(f"Dataset training harus memiliki kolom target `{TARGET_COL}`.")
        st.stop()

    unique_target = sorted(df[TARGET_COL].dropna().unique().tolist())
    if not set(unique_target).issubset({0, 1}):
        st.error(
            f"Kolom `{TARGET_COL}` harus berisi label 0/1 atau teks "
            f"'{NEGATIVE_LABEL_NAME}'/'{POSITIVE_LABEL_NAME}'. Nilai ditemukan: {unique_target}"
        )
        st.stop()

    if df[TARGET_COL].nunique() < 2:
        st.error("Data training harus memiliki minimal dua kelas target: Existing dan Attrited.")
        st.stop()

# =========================================================
# FUNGSI MODELING
# =========================================================
@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_resource(show_spinner=False)
def train_catboost_model(df: pd.DataFrame, test_size: float, threshold_mode: str):
    """Training model CatBoost dan evaluasi."""
    df = preprocess_training_data(df)
    check_dataset(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numerical_cols, categorical_cols = get_feature_types(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Split internal untuk early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    model = CatBoostClassifier(**DEFAULT_CATBOOST_PARAMS)
    model.fit(
        X_tr,
        y_tr,
        cat_features=categorical_cols,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )

    # Cari threshold optimal seperti notebook, fokus F2-score untuk recall churn
    y_val_prob = model.predict_proba(X_val)[:, 1]
    if threshold_mode == "Optimalkan F2-score":
        thresholds = np.arange(0.10, 0.91, 0.01)
        scores = [fbeta_score(y_val, (y_val_prob >= t).astype(int), beta=2) for t in thresholds]
        best_threshold = float(thresholds[int(np.argmax(scores))])
    else:
        best_threshold = 0.50

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "F2-Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=[NEGATIVE_LABEL_NAME, POSITIVE_LABEL_NAME],
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "threshold": best_threshold,
        "metrics": metrics,
        "report": report,
        "confusion_matrix": cm,
        "feature_columns": X.columns.tolist(),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "processed_df": df,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def predict_churn(model, df_input: pd.DataFrame, threshold: float):
    """Menghasilkan probabilitas dan label prediksi."""
    prob_churn = model.predict_proba(df_input)[:, 1]
    pred = (prob_churn >= threshold).astype(int)

    result = df_input.copy()
    result["Probabilitas_Churn"] = prob_churn
    result["Prediksi"] = np.where(pred == 1, POSITIVE_LABEL_NAME, NEGATIVE_LABEL_NAME)
    result["Risiko"] = pd.cut(
        prob_churn,
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Rendah", "Sedang", "Tinggi"]
    )
    return result

# =========================================================
# SIDEBAR - UPLOAD DAN SETTING
# =========================================================
st.sidebar.header("⚙️ Pengaturan")

uploaded_train = st.sidebar.file_uploader(
    "Upload dataset training CSV",
    type=["csv"],
    help=f"Dataset harus memiliki kolom target `{TARGET_COL}`."
)

test_size = st.sidebar.slider(
    "Proporsi data testing",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05,
)

threshold_mode = st.sidebar.selectbox(
    "Threshold prediksi",
    options=["Optimalkan F2-score", "Default 0.50"],
    index=0,
    help="F2-score memberi bobot lebih besar pada recall, cocok untuk mendeteksi nasabah churn."
)

train_button = st.sidebar.button("🚀 Latih Model", type="primary")

# =========================================================
# LOAD DATA
# =========================================================
if uploaded_train is None:
    st.info("Silakan upload file CSV training melalui sidebar. Contoh: BankChurners.csv")
    st.stop()

raw_df = load_csv(uploaded_train)
processed_preview = preprocess_training_data(raw_df)

st.subheader("1. Preview Data Setelah Preprocessing")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Baris", f"{processed_preview.shape[0]:,}")
col2.metric("Jumlah Kolom", f"{processed_preview.shape[1]:,}")
col3.metric("Jumlah Duplikasi", f"{processed_preview.duplicated().sum():,}")

st.dataframe(processed_preview.head(20), use_container_width=True)

if TARGET_COL in processed_preview.columns:
    st.write("Distribusi Target:")
    target_count = processed_preview[TARGET_COL].map({0: NEGATIVE_LABEL_NAME, 1: POSITIVE_LABEL_NAME}).value_counts()
    st.bar_chart(target_count)

# =========================================================
# TRAINING MODEL
# =========================================================
if train_button:
    with st.spinner("Melatih model CatBoost..."):
        st.session_state["model_result"] = train_catboost_model(raw_df, test_size, threshold_mode)

if "model_result" not in st.session_state:
    st.warning("Klik tombol **Latih Model** di sidebar untuk mulai training dan membuka menu prediksi.")
    st.stop()

result = st.session_state["model_result"]
model = result["model"]
threshold = result["threshold"]
feature_columns = result["feature_columns"]
numerical_cols = result["numerical_cols"]
categorical_cols = result["categorical_cols"]
processed_df = result["processed_df"]

st.success(f"Model berhasil dilatih. Threshold prediksi yang digunakan: **{threshold:.2f}**")

# =========================================================
# EVALUASI MODEL
# =========================================================
st.subheader("2. Evaluasi Model")
metrics_df = pd.DataFrame(result["metrics"], index=["CatBoost"]).T.reset_index()
metrics_df.columns = ["Metrik", "Nilai"]
st.dataframe(metrics_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.write("Confusion Matrix")
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

with c2:
    st.write("Feature Importance")
    fi = pd.DataFrame({
        "Fitur": feature_columns,
        "Importance": model.get_feature_importance()
    }).sort_values("Importance", ascending=False)
    st.dataframe(fi, use_container_width=True)
    st.bar_chart(fi.set_index("Fitur").head(15))

# =========================================================
# PREDIKSI
# =========================================================
st.subheader("3. Prediksi Customer Churn")

tab_manual, tab_csv = st.tabs(["Input Manual", "Upload CSV Prediksi"])

with tab_manual:
    st.write("Masukkan nilai fitur nasabah, lalu klik tombol prediksi.")

    input_data = {}
    n_cols = 3
    cols = st.columns(n_cols)

    for idx, col in enumerate(feature_columns):
        with cols[idx % n_cols]:
            if col in categorical_cols:
                options = sorted(processed_df[col].dropna().astype(str).unique().tolist())
                default_index = 0
                input_data[col] = st.selectbox(col, options=options, index=default_index)
            else:
                min_val = float(processed_df[col].min())
                max_val = float(processed_df[col].max())
                mean_val = float(processed_df[col].mean())

                # Gunakan number_input agar fleksibel untuk nilai di luar rentang data training
                if pd.api.types.is_integer_dtype(processed_df[col]):
                    input_data[col] = st.number_input(
                        col,
                        value=int(round(mean_val)),
                        step=1,
                    )
                else:
                    input_data[col] = st.number_input(
                        col,
                        value=round(mean_val, 3),
                        step=0.001,
                        format="%.3f",
                    )
                st.caption(f"Rentang data train: {min_val:.3f} - {max_val:.3f}")

    if st.button("🔍 Prediksi Input Manual", type="primary"):
        manual_df = pd.DataFrame([input_data])
        pred_result = predict_churn(model, manual_df[feature_columns], threshold)

        prob = float(pred_result.loc[0, "Probabilitas_Churn"])
        label = pred_result.loc[0, "Prediksi"]
        risk = pred_result.loc[0, "Risiko"]

        if label == POSITIVE_LABEL_NAME:
            st.error(f"Prediksi: **{label}** | Probabilitas churn: **{prob:.2%}** | Risiko: **{risk}**")
        else:
            st.success(f"Prediksi: **{label}** | Probabilitas churn: **{prob:.2%}** | Risiko: **{risk}**")

        st.dataframe(pred_result, use_container_width=True)

with tab_csv:
    st.write(
        "Upload CSV berisi fitur nasabah yang ingin diprediksi. "
        f"CSV boleh memiliki kolom `{TARGET_COL}`, tetapi kolom tersebut akan diabaikan."
    )

    uploaded_pred = st.file_uploader(
        "Upload CSV untuk prediksi",
        type=["csv"],
        key="prediction_csv"
    )

    if uploaded_pred is not None:
        pred_raw = pd.read_csv(uploaded_pred)
        st.write("Preview data prediksi:")
        st.dataframe(pred_raw.head(20), use_container_width=True)

        try:
            pred_df = preprocess_prediction_data(pred_raw, feature_columns)
            pred_result = predict_churn(model, pred_df, threshold)

            st.success("Prediksi berhasil dibuat.")
            st.dataframe(pred_result, use_container_width=True)

            csv_result = pred_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Hasil Prediksi CSV",
                data=csv_result,
                file_name="hasil_prediksi_churn.csv",
                mime="text/csv",
            )

            st.write("Ringkasan Prediksi:")
            st.bar_chart(pred_result["Prediksi"].value_counts())

        except Exception as e:
            st.error(f"Prediksi gagal: {e}")
            st.info(f"Kolom yang dibutuhkan: {feature_columns}")

# =========================================================
# INFORMASI KOLOM
# =========================================================
with st.expander("📌 Informasi Fitur yang Digunakan"):
    st.write("Fitur numerik:")
    st.write(numerical_cols)
    st.write("Fitur kategorikal:")
    st.write(categorical_cols)
    st.write("Kolom yang dibuang saat preprocessing:")
    st.write([col for col in COLS_TO_DROP if col in raw_df.columns])
