import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    roc_curve,
)

# ======================================================
# KONFIGURASI HALAMAN
# ======================================================
st.set_page_config(
    page_title="Prediksi Churn Kartu Kredit",
    page_icon="💳",
    layout="wide"
)

DATA_PATH = "data/BankChurners.csv"

TARGET_COL = "Attrition_Flag"
DROP_COLS = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

st.markdown(
    """
    <style>
    .main-header {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #666;
        font-size: 16px;
        margin-bottom: 20px;
    }
    .risk-low {
        background-color: #E8F5E9;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #43A047;
    }
    .risk-medium {
        background-color: #FFF8E1;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #FBC02D;
    }
    .risk-high {
        background-color: #FFEBEE;
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #E53935;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================================
# FUNGSI DATA
# ======================================================
@st.cache_data
def load_default_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def read_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Format file harus CSV atau Excel.")


def clean_dataset(df: pd.DataFrame, has_target: bool = True):
    """Membersihkan dataset sesuai notebook: drop CLIENTNUM dan kolom Naive Bayes."""
    data = df.copy()
    data = data.drop(columns=[c for c in DROP_COLS if c in data.columns], errors="ignore")

    if has_target and TARGET_COL in data.columns:
         target_as_text = data[TARGET_COL].astype(str).str.strip()
            mapped_target = target_as_text.map({
                "Existing Customer": 0,
                "Attrited Customer": 1,
                "Existing": 0,
                "Attrited": 1,
                "0": 0,
                "1": 1,
            })
        
            unknown_values = target_as_text[mapped_target.isna()].dropna().unique()
            if len(unknown_values) > 0:
                raise ValueError(
                    f"Nilai target {TARGET_COL} tidak dikenali: {list(unknown_values)}"
                )
        
            data[TARGET_COL] = mapped_target.astype(int)

    return data


def split_features_target(df_clean: pd.DataFrame):
    X = df_clean.drop(columns=[TARGET_COL])
    y = df_clean[TARGET_COL]
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, cat_features


def get_risk_category(prob):
    if prob < 0.30:
        return "Rendah"
    if prob < 0.60:
        return "Sedang"
    return "Tinggi"


def get_recommendation(risk):
    if risk == "Rendah":
        return "Nasabah memiliki risiko churn rendah. Pertahankan layanan, berikan promosi ringan, dan jaga engagement secara berkala."
    if risk == "Sedang":
        return "Nasabah perlu dipantau. Berikan penawaran cashback, reminder benefit kartu, atau program loyalitas sederhana."
    return "Nasabah berisiko tinggi churn. Prioritaskan untuk program retensi, follow-up customer care, evaluasi keluhan, dan tawarkan benefit khusus."


def validate_prediction_columns(input_df, feature_columns):
    missing = [c for c in feature_columns if c not in input_df.columns]
    if missing:
        return False, missing
    return True, []

# ======================================================
# FUNGSI MODEL
# ======================================================
def train_baseline_model(X_train, y_train, X_val, y_val, cat_features):
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        loss_function="Logloss",
        eval_metric="F1"
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    return model


def run_bayesian_optimization(X_train, y_train, cat_features, init_points=3, n_iter=10):
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as exc:
        raise ImportError(
            "Package bayesian-optimization belum terinstall. Jalankan: pip install bayesian-optimization"
        ) from exc

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def cv_f2(iterations, learning_rate, depth, l2_leaf_reg):
        iterations = int(round(iterations))
        depth = int(round(depth))
        fold_scores = []

        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
            )
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                use_best_model=True,
            )
            prob = model.predict_proba(X_va)[:, 1]
            pred = (prob >= 0.50).astype(int)
            fold_scores.append(fbeta_score(y_va, pred, beta=2, pos_label=1))

        return float(np.mean(fold_scores))

    optimizer = BayesianOptimization(
        f=cv_f2,
        pbounds={
            "iterations": (100, 500),
            "learning_rate": (0.01, 0.10),
            "depth": (3, 8),
            "l2_leaf_reg": (3.0, 50.0),
        },
        random_state=42,
        verbose=0,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    best_params = optimizer.max["params"]
    best_params["iterations"] = int(round(best_params["iterations"]))
    best_params["depth"] = int(round(best_params["depth"]))
    history = pd.DataFrame(optimizer.res)
    if not history.empty:
        params_df = pd.json_normalize(history["params"])
        history = pd.concat([history.drop(columns=["params"]), params_df], axis=1)
    return best_params, history


def find_best_threshold(X_train, y_train, cat_features, best_params):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_scores = []

    progress_bar = st.progress(0)
    status = st.empty()

    for i, t in enumerate(thresholds):
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model = CatBoostClassifier(
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=0,
                **best_params,
            )
            model.fit(
                X_tr,
                y_tr,
                cat_features=cat_features,
                eval_set=(X_va, y_va),
                early_stopping_rounds=50,
                use_best_model=True,
            )
            prob = model.predict_proba(X_va)[:, 1]
            pred = (prob >= t).astype(int)
            fold_scores.append(fbeta_score(y_va, pred, beta=2, pos_label=1))

        threshold_scores.append(float(np.mean(fold_scores)))
        progress_bar.progress((i + 1) / len(thresholds))
        status.write(f"Mencari threshold terbaik: {t:.2f}")

    best_idx = int(np.argmax(threshold_scores))
    best_threshold = float(thresholds[best_idx])
    result = pd.DataFrame({"threshold": thresholds, "f2_score": threshold_scores})
    status.empty()
    progress_bar.empty()
    return best_threshold, result


def train_optimized_model(X_tr_final, y_tr_final, X_val_final, y_val_final, cat_features, best_params):
    model = CatBoostClassifier(
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        **best_params,
    )
    model.fit(
        X_tr_final,
        y_tr_final,
        cat_features=cat_features,
        eval_set=(X_val_final, y_val_final),
        early_stopping_rounds=50,
        use_best_model=True,
    )
    return model


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "F2-Score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return metrics, y_pred, y_prob

def load_saved_model():
    return st.session_state.get("model"), st.session_state.get("metadata")

# ======================================================
# DATA DEFAULT
# ======================================================
raw_df = load_default_data()

with st.sidebar:
    st.title("💳 Churn App")
    menu = st.radio(
        "Menu",
        [
            "Dashboard",
            "Eksplorasi Data",
            "Training & Optimasi",
            "Evaluasi Model",
            "Prediksi Manual",
            "Prediksi Batch",
            "Tentang Sistem",
        ],
    )

    st.divider()
    st.caption("Dataset default")
    if raw_df is not None:
        st.success(f"Data tersedia: {raw_df.shape[0]} baris")
    else:
        st.warning("data/BankChurners.csv belum ditemukan")

# ======================================================
# HALAMAN DASHBOARD
# ======================================================
if menu == "Dashboard":
    st.markdown('<p class="main-header">Prediksi Churn Nasabah Kartu Kredit</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Implementasi CatBoost Classifier dengan Bayesian Optimization menggunakan Streamlit.</p>',
        unsafe_allow_html=True,
    )

    if raw_df is None:
        st.error("Dataset default belum ditemukan. Upload atau letakkan BankChurners.csv di folder data/.")
        st.stop()

    df_clean = clean_dataset(raw_df, has_target=True)
    churn_count = int((df_clean[TARGET_COL] == 1).sum())
    stay_count = int((df_clean[TARGET_COL] == 0).sum())
    churn_rate = churn_count / len(df_clean) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nasabah", f"{len(df_clean):,}")
    c2.metric("Nasabah Churn", f"{churn_count:,}")
    c3.metric("Nasabah Tidak Churn", f"{stay_count:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    label_map = {0: "Existing Customer", 1: "Attrited Customer"}
    plot_df = df_clean[TARGET_COL].map(label_map).value_counts().reset_index()
    plot_df.columns = ["Status", "Jumlah"]

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(plot_df, names="Status", values="Jumlah", title="Proporsi Churn")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.bar(plot_df, x="Status", y="Jumlah", text="Jumlah", title="Distribusi Target")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Preview Dataset")
    st.dataframe(raw_df.head(10), use_container_width=True)

# ======================================================
# HALAMAN EKSPLORASI DATA
# ======================================================
elif menu == "Eksplorasi Data":
    st.title("📊 Eksplorasi Data")
    if raw_df is None:
        st.error("Dataset default belum ditemukan.")
        st.stop()

    df_clean = clean_dataset(raw_df, has_target=True)
    st.write("Ukuran data setelah seleksi atribut:", df_clean.shape)

    tab1, tab2, tab3, tab4 = st.tabs(["Struktur Data", "Missing & Duplikat", "Visualisasi", "Korelasi"])

    with tab1:
        st.subheader("Kolom Dataset")
        info_df = pd.DataFrame({
            "Kolom": df_clean.columns,
            "Tipe Data": [str(df_clean[c].dtype) for c in df_clean.columns],
            "Jumlah Unik": [df_clean[c].nunique() for c in df_clean.columns],
        })
        st.dataframe(info_df, use_container_width=True)
        st.subheader("Statistik Deskriptif")
        st.dataframe(df_clean.describe().T, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Missing Value", int(df_clean.isnull().sum().sum()))
            st.dataframe(df_clean.isnull().sum().reset_index().rename(columns={"index": "Kolom", 0: "Missing"}))
        with col2:
            st.metric("Data Duplikat", int(df_clean.duplicated().sum()))
            unknowns = []
            for c in df_clean.select_dtypes(include="object").columns:
                unknowns.append({"Kolom": c, "Jumlah Unknown": int((df_clean[c] == "Unknown").sum())})
            st.dataframe(pd.DataFrame(unknowns), use_container_width=True)

    with tab3:
        target_label = df_clean[TARGET_COL].map({0: "Existing Customer", 1: "Attrited Customer"})
        temp = df_clean.copy()
        temp["Status"] = target_label
        num_cols = temp.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET_COL, errors="ignore").tolist()
        cat_cols = temp.select_dtypes(include=["object", "category"]).columns.tolist()

        selected_num = st.selectbox("Pilih fitur numerik", num_cols)
        fig = px.histogram(temp, x=selected_num, color="Status", marginal="box", title=f"Distribusi {selected_num} terhadap Churn")
        st.plotly_chart(fig, use_container_width=True)

        selected_cat = st.selectbox("Pilih fitur kategorikal", cat_cols)
        cat_plot = temp.groupby([selected_cat, "Status"]).size().reset_index(name="Jumlah")
        fig = px.bar(cat_plot, x=selected_cat, y="Jumlah", color="Status", barmode="group", title=f"{selected_cat} terhadap Churn")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.drop(TARGET_COL, errors="ignore").tolist()
        corr = df_clean[num_cols].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Heatmap Korelasi Fitur Numerik")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# HALAMAN TRAINING DAN OPTIMASI
# ======================================================
elif menu == "Training & Optimasi":
    st.title("🤖 Training CatBoost + Bayesian Optimization")
    if raw_df is None:
        st.error("Dataset default belum ditemukan.")
        st.stop()

    st.info("Halaman ini mengikuti alur notebook: seleksi atribut, transformasi target, train-test split, CatBoost baseline, Bayesian Optimization, optimasi threshold, lalu training model final.")

    with st.expander("Pengaturan training", expanded=True):
        test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
        val_size = st.slider("Validation size dari data training", 0.10, 0.40, 0.20, 0.05)
        init_points = st.number_input("Bayesian init_points", 1, 10, 3)
        n_iter = st.number_input("Bayesian n_iter", 1, 30, 10)
        run_threshold_search = st.checkbox("Cari threshold optimal berdasarkan F2-score", value=True)

    if st.button("🚀 Jalankan Training & Optimasi", type="primary"):
        start_time = time.time()
        df_clean = clean_dataset(raw_df, has_target=True)
        X, y, cat_features = split_features_target(df_clean)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
        )

        st.write("Fitur kategorikal untuk CatBoost:", cat_features)
        st.write("Ukuran data train/test:", X_train.shape, X_test.shape)

        with st.spinner("Training CatBoost baseline..."):
            baseline_model = train_baseline_model(X_tr_final, y_tr_final, X_val_final, y_val_final, cat_features)
            baseline_metrics, y_pred_base, y_prob_base = evaluate_model(baseline_model, X_test, y_test, threshold=0.5)

        st.success("Baseline selesai.")
        st.subheader("Metrik Baseline")
        st.dataframe(pd.DataFrame([baseline_metrics]).T.rename(columns={0: "Score"}), use_container_width=True)

        with st.spinner("Menjalankan Bayesian Optimization..."):
            best_params, bo_history = run_bayesian_optimization(
                X_train, y_train, cat_features, init_points=int(init_points), n_iter=int(n_iter)
            )

        st.success("Bayesian Optimization selesai.")
        st.subheader("Best Parameter")
        st.json(best_params)

        if not bo_history.empty:
            st.subheader("Riwayat Bayesian Optimization")
            st.dataframe(bo_history, use_container_width=True)
            fig = px.line(bo_history.reset_index(), x="index", y="target", markers=True, title="Perkembangan Skor F2 pada Bayesian Optimization")
            st.plotly_chart(fig, use_container_width=True)

        if run_threshold_search:
            st.subheader("Optimasi Threshold")
            best_threshold, threshold_df = find_best_threshold(X_train, y_train, cat_features, best_params)
            st.success(f"Threshold terbaik: {best_threshold:.2f}")
            fig = px.line(threshold_df, x="threshold", y="f2_score", title="F2-score Berdasarkan Threshold")
            st.plotly_chart(fig, use_container_width=True)
        else:
            best_threshold = 0.50
            threshold_df = pd.DataFrame()

        with st.spinner("Training model final CatBoost + Bayesian Optimization..."):
            bo_model = train_optimized_model(X_tr_final, y_tr_final, X_val_final, y_val_final, cat_features, best_params)
            bo_metrics, y_pred_bo, y_prob_bo = evaluate_model(bo_model, X_test, y_test, threshold=best_threshold)

        compare_df = pd.DataFrame([
            {"Model": "CatBoost Baseline", **baseline_metrics},
            {"Model": "CatBoost + Bayesian Optimization", **bo_metrics},
        ])

        metadata = {
            "feature_columns": X.columns.tolist(),
            "cat_features": cat_features,
            "best_params": best_params,
            "best_threshold": best_threshold,
            "baseline_metrics": baseline_metrics,
            "bo_metrics": bo_metrics,
            "comparison": compare_df,
            "classification_report": classification_report(y_test, y_pred_bo, target_names=["Tidak Churn", "Churn"], output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred_bo).tolist(),
            "y_test": y_test.to_numpy(),
            "y_pred_bo": y_pred_bo,
            "y_prob_bo": y_prob_bo,
            "training_time_seconds": time.time() - start_time,
        }
        st.session_state["model"] = bo_model
        st.session_state["metadata"] = metadata

        st.session_state["model"] = bo_model
        st.session_state["metadata"] = metadata

        st.success("Model final berhasil dilatih/.")
        st.subheader("Perbandingan Model")
        st.dataframe(compare_df, use_container_width=True)

        fig = px.bar(
            compare_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Perbandingan CatBoost Baseline vs CatBoost + Bayesian Optimization",
        )
        st.plotly_chart(fig, use_container_width=True)

# ======================================================
# HALAMAN EVALUASI MODEL
# ======================================================
elif menu == "Evaluasi Model":
    st.title("📈 Evaluasi Model")
    model, metadata = load_saved_model()
    if metadata is None:
        st.warning("Model belum ditemukan. Jalankan training terlebih dahulu di menu Training & Optimasi.")
        st.stop()

    st.subheader("Parameter Terbaik")
    st.json(metadata["best_params"])
    st.metric("Threshold Terbaik", f"{metadata['best_threshold']:.2f}")

    compare_df = metadata["comparison"]
    st.subheader("Perbandingan Metrik")
    st.dataframe(compare_df, use_container_width=True)

    fig = px.bar(
        compare_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Metrik Evaluasi Model",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = np.array(metadata["confusion_matrix"])
        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Pred Tidak Churn", "Pred Churn"],
            y=["Aktual Tidak Churn", "Aktual Churn"],
            title="Confusion Matrix CatBoost + BO",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("ROC Curve")
        y_test = np.array(metadata["y_test"])
        y_prob = np.array(metadata["y_prob_bo"])
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {roc_auc:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    feature_columns = metadata["feature_columns"]
    importance = model.get_feature_importance()
    imp_df = pd.DataFrame({"Feature": feature_columns, "Importance": importance}).sort_values("Importance", ascending=False)
    fig = px.bar(imp_df.head(15), x="Importance", y="Feature", orientation="h", title="Top 15 Feature Importance")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(imp_df, use_container_width=True)

# ======================================================
# HALAMAN PREDIKSI MANUAL
# ======================================================
elif menu == "Prediksi Manual":
    st.title("🎯 Prediksi Manual Satu Nasabah")
    model, metadata = load_saved_model()
    if metadata is None:
        st.warning("Model belum ditemukan. Jalankan training terlebih dahulu di menu Training & Optimasi.")
        st.stop()

    if raw_df is None:
        st.error("Dataset default belum tersedia untuk mengambil pilihan kategori.")
        st.stop()

    df_clean = clean_dataset(raw_df, has_target=True)
    feature_columns = metadata["feature_columns"]
    cat_features = metadata["cat_features"]
    threshold = metadata["best_threshold"]

    st.write("Isi data nasabah berikut, lalu klik tombol prediksi.")
    input_data = {}

    col1, col2, col3 = st.columns(3)
    for i, col in enumerate(feature_columns):
        target_col_area = [col1, col2, col3][i % 3]
        with target_col_area:
            if col in cat_features:
                options = sorted(df_clean[col].dropna().astype(str).unique().tolist())
                input_data[col] = st.selectbox(col, options=options)
            else:
                min_val = float(df_clean[col].min())
                max_val = float(df_clean[col].max())
                mean_val = float(df_clean[col].mean())
                if pd.api.types.is_integer_dtype(df_clean[col]):
                    input_data[col] = st.number_input(col, min_value=int(min_val), max_value=int(max_val), value=int(round(mean_val)))
                else:
                    input_data[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val, format="%.3f")

    if st.button("Prediksi Churn", type="primary"):
        input_df = pd.DataFrame([input_data])[feature_columns]
        prob = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(prob >= threshold)
        risk = get_risk_category(prob)
        recommendation = get_recommendation(risk)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediksi", "Churn" if pred == 1 else "Tidak Churn")
        c2.metric("Probabilitas Churn", f"{prob * 100:.2f}%")
        c3.metric("Kategori Risiko", risk)

        css_class = "risk-low" if risk == "Rendah" else "risk-medium" if risk == "Sedang" else "risk-high"
        st.markdown(
            f"<div class='{css_class}'><b>Rekomendasi:</b><br>{recommendation}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Data Input")
        st.dataframe(input_df, use_container_width=True)

# ======================================================
# HALAMAN PREDIKSI BATCH
# ======================================================
elif menu == "Prediksi Batch":
    st.title("📤 Prediksi Batch")
    model, metadata = load_saved_model()
    if metadata is None:
        st.warning("Model belum ditemukan. Jalankan training terlebih dahulu di menu Training & Optimasi.")
        st.stop()

    uploaded = st.file_uploader("Upload data nasabah baru CSV/XLSX", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        batch_raw = read_uploaded_file(uploaded)
        st.subheader("Preview Data Upload")
        st.dataframe(batch_raw.head(), use_container_width=True)

        feature_columns = metadata["feature_columns"]
        threshold = metadata["best_threshold"]
        batch_clean = clean_dataset(batch_raw, has_target=False)

        valid, missing = validate_prediction_columns(batch_clean, feature_columns)
        if not valid:
            st.error("Kolom input belum lengkap. Kolom yang hilang:")
            st.write(missing)
            st.stop()

        batch_X = batch_clean[feature_columns]
        probs = model.predict_proba(batch_X)[:, 1]
        preds = (probs >= threshold).astype(int)

        result = batch_raw.copy()
        result["Probabilitas_Churn"] = probs
        result["Probabilitas_Churn_%"] = probs * 100
        result["Prediksi"] = np.where(preds == 1, "Churn", "Tidak Churn")
        result["Kategori_Risiko"] = [get_risk_category(p) for p in probs]
        result["Rekomendasi"] = [get_recommendation(r) for r in result["Kategori_Risiko"]]

        st.subheader("Hasil Prediksi")
        st.dataframe(result, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", len(result))
        c2.metric("Prediksi Churn", int((result["Prediksi"] == "Churn").sum()))
        c3.metric("Rata-rata Probabilitas Churn", f"{result['Probabilitas_Churn_%'].mean():.2f}%")

        csv_buffer = io.StringIO()
        result.to_csv(csv_buffer, index=False)
        st.download_button(
            "Download Hasil Prediksi CSV",
            data=csv_buffer.getvalue(),
            file_name="hasil_prediksi_churn.csv",
            mime="text/csv",
        )

# ======================================================
# HALAMAN TENTANG
# ======================================================
elif menu == "Tentang Sistem":
    st.title("ℹ️ Tentang Sistem")
    st.markdown(
        """
        Sistem ini dibuat untuk memprediksi kemungkinan churn nasabah kartu kredit.

        **Metode utama:**
        - CatBoost Classifier
        - Bayesian Optimization untuk optimasi hyperparameter
        - F2-score sebagai fungsi objektif optimasi karena kasus churn lebih menekankan recall
        - Threshold optimal untuk menentukan batas probabilitas churn

        **Tahapan:**
        1. Seleksi atribut
        2. Transformasi target
        3. Split data train-test
        4. Training CatBoost baseline
        5. Optimasi Bayesian
        6. Optimasi threshold
        7. Evaluasi model
        8. Prediksi manual dan batch

        **Mapping target:**
        - Existing Customer = 0 / Tidak Churn
        - Attrited Customer = 1 / Churn
        """
    )
