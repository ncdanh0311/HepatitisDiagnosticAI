# 🏥 Hepatitis Disease Analysis & Prediction

Hệ thống phân tích và dự đoán bệnh viêm gan từ chỉ số sinh hóa máu.
Kết hợp **EDA · Machine Learning · NLP Voice Assistant** trên giao diện Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=flat-square)

---

## 🚀 Chạy ngay

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Cấu trúc

```
hepatitis-eda/
├── app.py                  # Trang chủ Streamlit
├── config.py               # Centralized config (paths, hyperparams)
├── requirements.txt
│
├── src/
│   ├── data/
│   │   └── processor.py    # DataProcessor — đọc, impute, scale, PCA/KBest
│   ├── models/
│   │   └── trainer.py      # ModelTrainer — RF · XGBoost · SVM + joblib cache
│   └── nlp/
│       └── intent_classifier.py  # TF-IDF + LogReg intent classifier
│
├── pages/
│   ├── 1_📊_EDA.py         # Phân tích dữ liệu
│   ├── 2_🤖_Model_Training.py  # Huấn luyện & đánh giá
│   ├── 3_🩺_Prediction.py  # Dự đoán ca bệnh mới
│   └── 4_🧠_NLP_Demo.py    # Thử intent classifier
│
├── data/
│   └── hepatitis.csv       # UCI HCV dataset (615 bệnh nhân)
├── models/                 # Saved .pkl (auto-generated sau lần train đầu)
└── .streamlit/
    └── config.toml         # Dark theme
```

---

## 🧪 Dataset

| | |
|---|---|
| **Nguồn** | [UCI ML Repository — HCV Data](https://archive.ics.uci.edu/ml/datasets/HCV+data) |
| **Kích thước** | 615 bệnh nhân × 12 chỉ số sinh hóa |
| **Target** | Category (5 lớp: Blood Donor → Cirrhosis) |
| **Missing** | ALB, ALP, ALT, CHOL, PROT, GGT |

---

## 🤖 ML Pipeline

| Bước | Kỹ thuật |
|------|---------|
| Missing data | `IterativeImputer` (MICE) |
| Imbalanced data | `SMOTE` (adaptive k_neighbors) + `class_weight='balanced'` |
| Models | Random Forest · XGBoost · SVM (RBF) |
| Evaluation | Stratified 5-Fold CV · Classification Report · Confusion Matrix |
| Persistence | `joblib` — model được lưu sau lần train đầu, load lại tự động |

---

## 🧠 NLP Intent Classifier

Thay thế hệ thống `if-else` cứng nhắc bằng mô hình ML:

```
Input text → TF-IDF (char n-gram 2–4) → Logistic Regression → Intent + Confidence
```

Hỗ trợ **22 intents** · Hiển thị top-5 xác suất · Confidence threshold có thể điều chỉnh.

---

## 📦 Dependencies

```
numpy · pandas · scipy
scikit-learn · xgboost · imbalanced-learn
streamlit · plotly
joblib
```

---

**Tác giả:** Đặng Nguyễn Quang Huy
