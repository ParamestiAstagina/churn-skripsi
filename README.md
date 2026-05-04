# Streamlit Prediksi Churn Nasabah Kartu Kredit

Aplikasi ini menggunakan CatBoost Classifier yang dioptimasi dengan Bayesian Optimization untuk memprediksi churn nasabah kartu kredit.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Letakkan file `BankChurners.csv` di folder `data/`.

## Fitur aplikasi
- Dashboard ringkasan data
- Eksplorasi data
- Training CatBoost baseline
- Optimasi hyperparameter CatBoost dengan Bayesian Optimization
- Optimasi threshold berdasarkan F2-score
- Evaluasi model
- Prediksi manual satu nasabah
- Prediksi batch dari file CSV/XLSX
- Feature importance
- Download hasil prediksi
