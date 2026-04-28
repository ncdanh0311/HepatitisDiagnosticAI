# 🏥 Hepatitis Disease Analysis & Prediction System

> **Hệ thống phân tích và dự đoán bệnh viêm gan dựa trên chỉ số sinh hóa**
>
> Kết hợp **Xử lý dữ liệu (EDA)** + **Machine Learning** + **NLP Voice Assistant** trong một ứng dụng desktop tích hợp.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

---

## 📌 Motivation (Tại sao làm dự án này?)

Viêm gan là một trong những bệnh lý nghiêm trọng ảnh hưởng đến hàng triệu người trên thế giới. Việc **phát hiện sớm** giai đoạn bệnh (Hepatitis → Fibrosis → Cirrhosis) thông qua các **chỉ số sinh hóa máu** có thể cứu sống nhiều bệnh nhân.

Dự án này được xây dựng với mục tiêu:

1. **Phân tích dữ liệu (EDA)** — Khám phá mối quan hệ giữa các chỉ số sinh hóa và giai đoạn bệnh
2. **Dự đoán bệnh (ML)** — Huấn luyện mô hình AI dự đoán tình trạng bệnh từ xét nghiệm máu
3. **Trợ lý giọng nói (NLP)** — Tương tác với hệ thống bằng tiếng Việt tự nhiên
4. **Thực hành kỹ thuật** — Áp dụng các kỹ thuật xử lý missing data, imbalanced data, và feature engineering

---

## 📊 Dataset (Nguồn dữ liệu)

| Thông tin | Chi tiết |
|-----------|----------|
| **Nguồn** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HCV+data) |
| **Số mẫu** | 615 bệnh nhân |
| **Số features** | 12 chỉ số sinh hóa + Tuổi + Giới tính |
| **Target** | Category (5 lớp: Blood Donor, Suspect, Hepatitis, Fibrosis, Cirrhosis) |
| **Missing Values** | Có (ALB, ALP, ALT, CHOL, PROT, GGT) |

### Các chỉ số sinh hóa (Features):

| Viết tắt | Tên đầy đủ | Ý nghĩa |
|----------|------------|---------|
| ALB | Albumin | Protein do gan sản xuất |
| ALP | Alkaline Phosphatase | Enzyme đánh giá chức năng gan-mật |
| ALT | Alanine Aminotransferase | Enzyme chỉ thị tổn thương gan |
| AST | Aspartate Aminotransferase | Enzyme chỉ thị tổn thương tế bào |
| BIL | Bilirubin | Sắc tố mật, đánh giá chức năng gan |
| CHE | Cholinesterase | Enzyme đánh giá khả năng tổng hợp của gan |
| CHOL | Cholesterol | Chỉ số mỡ máu |
| CREA | Creatinine | Chỉ số chức năng thận |
| GGT | Gamma-Glutamyl Transferase | Enzyme đánh giá tổn thương gan-mật |
| PROT | Total Proteins | Tổng protein huyết thanh |

---

## 🏗️ Architecture (Kiến trúc hệ thống)

```
┌─────────────────────────────────────────────────────────────────┐
│                    HEPATITIS EDA SYSTEM                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  🎤 Voice │───▶│ Mono         │───▶│ NLP IntentClassifier  │  │
│  │  Input    │    │ (STT Engine) │    │ TF-IDF + LogReg       │  │
│  └──────────┘    └──────────────┘    └───────┬───────────────┘  │
│                                              │                   │
│  ┌──────────┐                                │  Intent           │
│  │  ⌨️ Text  │────────────────────────────────┤                   │
│  │  Input    │                                ▼                   │
│  └──────────┘                    ┌────────────────────────┐      │
│                                  │  Practice Controller   │      │
│                                  │  (Command Dispatcher)  │      │
│                                  └───────┬────────────────┘      │
│                                          │                       │
│              ┌───────────────────────────┼──────────────────┐    │
│              ▼                           ▼                  ▼    │
│   ┌──────────────────┐    ┌──────────────────┐  ┌───────────┐   │
│   │  DataProcessor   │    │  ModelTrainer     │  │  Charts   │   │
│   │  ─────────────   │    │  ──────────────   │  │  ───────  │   │
│   │  • Read CSV      │    │  • IterativeImp.  │  │  • Line   │   │
│   │  • Clean Data    │    │  • SMOTE          │  │  • Scatter│   │
│   │  • Z-Score       │    │  • RandomForest   │  │  • Bar    │   │
│   │  • MinMax Scale  │    │  • XGBoost        │  │  • Heatmap│   │
│   │  • PCA / KBest   │    │  • SVM            │  │           │   │
│   └──────────────────┘    └──────────────────┘  └───────────┘   │
│              │                       │                  │        │
│              └───────────────────────┼──────────────────┘        │
│                                      ▼                           │
│                          ┌─────────────────────┐                 │
│                          │   Tkinter UI         │                 │
│                          │   ─────────────      │                 │
│                          │   • Treeview Table   │                 │
│                          │   • ScrolledText     │                 │
│                          │   • Matplotlib Canvas│                 │
│                          │   • Menu Bar         │                 │
│                          └─────────────────────┘                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Luồng dữ liệu chi tiết:

```
hepatitis.csv ──▶ DataProcessor ──▶ IterativeImputer ──▶ StandardScaler
                                                              │
                                    ┌─────────────────────────┤
                                    ▼                         ▼
                              SMOTE (Oversampling)     Train/Test Split
                                    │                         │
                                    ▼                         ▼
                          ┌─────────────────┐    ┌──────────────────┐
                          │  Training Data  │    │   Test Data      │
                          └────────┬────────┘    └────────┬─────────┘
                                   │                      │
                    ┌──────────────┼──────────────┐       │
                    ▼              ▼               ▼      │
              RandomForest    XGBoost           SVM       │
                    │              │               │      │
                    └──────────────┼───────────────┘      │
                                   ▼                      │
                          Cross-Validation ◀──────────────┘
                                   │
                                   ▼
                    Classification Report + Confusion Matrix
```

---

## 🚀 Getting Started (Hướng dẫn cài đặt)

### Yêu cầu

- Python 3.10 trở lên
- Microphone (cho tính năng voice command)

### Cài đặt

```bash
# 1. Clone repository
git clone https://github.com/huydeptrai1/DOANPYTHON.git
cd DOANPYTHON

# 2. Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Chạy ứng dụng
python main.py
```

---

## 📂 Project Structure (Cấu trúc dự án)

```
hepatitis-eda/
│
├── main.py                 # Entry point - Khởi chạy ứng dụng
├── UI.py                   # Giao diện chính (Tkinter)
├── UI_SimpleDialog.py      # Dialog nhập lệnh bàn phím
│
├── Process.py              # DataProcessor + UIController (refactored)
├── Process_Chart.py        # Module vẽ biểu đồ (Matplotlib)
├── Practice.py             # Command dispatcher (NLP-powered)
│
├── trainer.py              # 🤖 ML Training Pipeline
├── nlp_intent.py           # 🧠 NLP Intent Classification
├── Mono.py                 # 🎤 Voice Assistant (TTS + STT)
│
├── Game.py                 # 🎮 Mini-game (Pygame)
├── hepatitis.csv           # 📊 Dataset
├── requirements.txt        # 📦 Dependencies
├── README.md               # 📖 Tài liệu dự án
│
└── assets/                 # Resources
    ├── app.ico
    ├── background.png
    ├── car.png
    └── obstacles.png
```

---

## 🤖 Module 1: Model Training (`trainer.py`)

### Kỹ thuật xử lý dữ liệu

| Vấn đề | Giải pháp | Chi tiết |
|---------|-----------|----------|
| **Missing Data** | `IterativeImputer` (MICE) | Dự đoán giá trị thiếu dựa trên các cột khác, thay vì xóa bỏ |
| **Imbalanced Data** | `SMOTE` + `class_weight` | Oversampling lớp thiểu số + điều chỉnh trọng số |
| **Feature Scaling** | `StandardScaler` | Chuẩn hóa Z-score cho SVM và các model nhạy với scale |

### Các mô hình

| Model | Đặc điểm | Hyperparameters |
|-------|----------|-----------------|
| **Random Forest** | Ensemble, robust với noise | `n_estimators=200, max_depth=15` |
| **XGBoost** | Gradient Boosting, hiệu suất cao | `learning_rate=0.1, subsample=0.8` |
| **SVM (RBF)** | Kernel trick, tốt cho non-linear | `C=10, gamma='scale'` |

### Chạy riêng module training

```bash
python trainer.py
```

---

## 🧠 Module 2: NLP Intent Classification (`nlp_intent.py`)

### So sánh: if-else vs NLP

| Tiêu chí | if-else (cũ) | NLP (mới) |
|----------|--------------|-----------|
| **Đầu vào** | Phải khớp chính xác từ khóa | Hiểu nhiều biến thể câu |
| **Mở rộng** | Sửa code, thêm if-else | Chỉ thêm training data |
| **Confidence** | Không có | Trả về % tin cậy |
| **Typo-tolerant** | Không | Có (nhờ char n-gram) |

### Ví dụ hoạt động

```
Input:  "Vẽ cho tôi cái biểu đồ đường"
Output: Intent = draw_line_chart (95% confidence)

Input:  "xem dữ liệu đi"
Output: Intent = load_data (92% confidence)

Input:  "chạy machine learning"
Output: Intent = train_model (88% confidence)
```

### Chạy demo NLP

```bash
python nlp_intent.py
```

---

## 📋 Module 3: Code Architecture (Kiến trúc Code)

### Nguyên tắc thiết kế

- **Separation of Concerns**: `DataProcessor` (logic) tách khỏi `UIController` (UI)
- **Single Responsibility**: Mỗi class/module chỉ làm một việc
- **Backward Compatibility**: Code cũ vẫn hoạt động qua alias `Process = UIController`
- **Dependency Injection**: UI components được truyền qua constructor

### Class Diagram

```
DataProcessor (Pure Logic)
├── _read_file()
├── check_null()
├── describe()
├── fill_missing_with_mode()
├── drop_null_rows()
├── minmax_scale()
├── compute_zscore()
├── remove_outliers()
├── extract_features_pca()
└── extract_features_kbest()

UIController (UI Bridge)
├── processor: DataProcessor
├── PrintData_Clicked()
├── PrintDescribe()
├── ...
└── ChoosefiltColumnsPCA_Clicked()

Chart(UIController)
├── drawchartLine_Clicked()
├── drawchartsca_Clicked()
└── ...

Practice(Chart)
├── classifier: IntentClassifier
├── PrintResult_Clicked()  ← NLP-powered
├── TrainModel_Clicked()   ← ML pipeline
└── ...
```

---

## 👤 Author

**Đặng Nguyễn Quang Huy**

---

## 📄 License

This project is for educational purposes.
