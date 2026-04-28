"""
Module NLP Intent Classification
=================================
Thay thế hệ thống if-else cứng nhắc bằng mô hình ML phân loại Intent.

Kỹ thuật:
- TfidfVectorizer: Chuyển văn bản thành vector số (TF-IDF)
- LogisticRegression: Phân loại intent từ vector
- Augmented Training Data: Nhiều biến thể câu lệnh tiếng Việt

Pipeline: Input text → TfidfVectorizer → LogisticRegression → Intent label

"""

import json
import pickle
from pathlib import Path
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "intent_model.pkl"


def configure_console_output():
    """Make console output safer on Windows terminals with legacy encodings."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass


configure_console_output()


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA: Vietnamese command phrases → Intent labels
# ═══════════════════════════════════════════════════════════════════════════════

TRAINING_DATA = [
    # ── load_data ─────────────────────────────────────────────
    ("dữ liệu", "load_data"),
    ("hiển thị dữ liệu", "load_data"),
    ("xem dữ liệu", "load_data"),
    ("tải dữ liệu", "load_data"),
    ("đọc dữ liệu", "load_data"),
    ("mở dữ liệu", "load_data"),
    ("load data", "load_data"),
    ("cho tôi xem dữ liệu", "load_data"),
    ("in dữ liệu ra", "load_data"),
    ("show data", "load_data"),
    ("hiện dữ liệu lên", "load_data"),

    # ── remove_columns ────────────────────────────────────────
    ("cột", "remove_columns"),
    ("xóa cột", "remove_columns"),
    ("loại bỏ cột", "remove_columns"),
    ("bỏ cột không cần", "remove_columns"),
    ("remove columns", "remove_columns"),
    ("xóa các cột không quan trọng", "remove_columns"),
    ("loại cột thừa", "remove_columns"),

    # ── replace_row ───────────────────────────────────────────
    ("thay thế hàng", "replace_row"),
    ("thay thế giá trị", "replace_row"),
    ("điền giá trị thiếu", "replace_row"),
    ("fill missing values", "replace_row"),
    ("thay thế null", "replace_row"),
    ("điền giá trị phổ biến", "replace_row"),
    ("thay thế NA", "replace_row"),

    # ── remove_row ────────────────────────────────────────────
    ("hàng", "remove_row"),
    ("xóa hàng", "remove_row"),
    ("loại bỏ hàng", "remove_row"),
    ("xóa dòng null", "remove_row"),
    ("remove rows", "remove_row"),
    ("xóa các dòng trống", "remove_row"),
    ("loại bỏ dòng rỗng", "remove_row"),

    # ── describe ──────────────────────────────────────────────
    ("lấy mô tả", "describe"),
    ("mô tả dữ liệu", "describe"),
    ("thống kê mô tả", "describe"),
    ("describe", "describe"),
    ("tóm tắt dữ liệu", "describe"),
    ("xem thống kê", "describe"),
    ("chi tiết dữ liệu", "describe"),

    # ── check_null ────────────────────────────────────────────
    ("kiểm tra rỗng", "check_null"),
    ("kiểm tra null", "check_null"),
    ("tìm giá trị null", "check_null"),
    ("check null", "check_null"),
    ("xem giá trị thiếu", "check_null"),
    ("kiểm tra missing", "check_null"),
    ("có bao nhiêu null", "check_null"),

    # ── find_popular ──────────────────────────────────────────
    ("tìm giá trị phổ biến", "find_popular"),
    ("giá trị phổ biến", "find_popular"),
    ("mode", "find_popular"),
    ("giá trị xuất hiện nhiều nhất", "find_popular"),
    ("tìm mode", "find_popular"),
    ("giá trị thường gặp", "find_popular"),

    # ── reload ────────────────────────────────────────────────
    ("tạo mới", "reload"),
    ("reset", "reload"),
    ("làm mới", "reload"),
    ("tải lại", "reload"),
    ("reload", "reload"),
    ("khởi tạo lại", "reload"),
    ("bắt đầu lại", "reload"),

    # ── remove_isolated ───────────────────────────────────────
    ("giá trị cá biệt", "remove_isolated"),
    ("loại bỏ outlier", "remove_isolated"),
    ("xóa outlier", "remove_isolated"),
    ("remove outliers", "remove_isolated"),
    ("xóa giá trị ngoại lai", "remove_isolated"),
    ("loại bỏ ngoại lai", "remove_isolated"),
    ("xử lý outlier", "remove_isolated"),

    # ── clear ─────────────────────────────────────────────────
    ("nội dung", "clear"),
    ("xóa nội dung", "clear"),
    ("xóa màn hình", "clear"),
    ("clear", "clear"),
    ("làm sạch", "clear"),
    ("dọn dẹp", "clear"),
    ("xóa hết", "clear"),

    # ── draw_line_chart ───────────────────────────────────────
    ("vẽ biểu đồ đường", "draw_line_chart"),
    ("biểu đồ đường", "draw_line_chart"),
    ("line chart", "draw_line_chart"),
    ("vẽ đường", "draw_line_chart"),
    ("draw line", "draw_line_chart"),
    ("biểu đồ line", "draw_line_chart"),

    # ── print_zscore ──────────────────────────────────────────
    ("xuất ma trận", "print_zscore"),
    ("ma trận zscore", "print_zscore"),
    ("tính zscore", "print_zscore"),
    ("z-score", "print_zscore"),
    ("xuất ma trận zscore", "print_zscore"),
    ("hiển thị zscore", "print_zscore"),

    # ── draw_scatter ──────────────────────────────────────────
    ("vẽ biểu đồ khác", "draw_scatter"),
    ("biểu đồ scatter", "draw_scatter"),
    ("scatter plot", "draw_scatter"),
    ("vẽ phân tán", "draw_scatter"),
    ("biểu đồ phân tán", "draw_scatter"),

    # ── minmax_scaler ─────────────────────────────────────────
    ("chuẩn hóa rời rạc", "minmax_scaler"),
    ("minmax", "minmax_scaler"),
    ("chuẩn hóa", "minmax_scaler"),
    ("normalize", "minmax_scaler"),
    ("min max scaler", "minmax_scaler"),
    ("chuẩn hóa dữ liệu", "minmax_scaler"),

    # ── filter_kbest ──────────────────────────────────────────
    ("trích lọc k", "filter_kbest"),
    ("kbest", "filter_kbest"),
    ("chọn k thuộc tính tốt nhất", "filter_kbest"),
    ("select k best", "filter_kbest"),
    ("trích lọc thuộc tính k", "filter_kbest"),
    ("lọc kbest", "filter_kbest"),

    # ── filter_pca ────────────────────────────────────────────
    ("trích lọc p", "filter_pca"),
    ("pca", "filter_pca"),
    ("giảm chiều", "filter_pca"),
    ("principal component", "filter_pca"),
    ("trích lọc pca", "filter_pca"),
    ("phân tích thành phần chính", "filter_pca"),

    # ── show_features ─────────────────────────────────────────
    ("hiển thị kết quả", "show_features"),
    ("xem kết quả", "show_features"),
    ("kết quả trích lọc", "show_features"),
    ("show results", "show_features"),
    ("hiện kết quả", "show_features"),
    ("in kết quả", "show_features"),

    # ── delete_chart ──────────────────────────────────────────
    ("xóa biểu đồ", "delete_chart"),
    ("xóa chart", "delete_chart"),
    ("delete chart", "delete_chart"),
    ("clear chart", "delete_chart"),
    ("xóa đồ thị", "delete_chart"),
    ("dọn biểu đồ", "delete_chart"),

    # ── draw_feature_scatter ──────────────────────────────────
    ("vẽ cột đặc trưng", "draw_feature_scatter"),
    ("vẽ thuộc tính đặc trưng", "draw_feature_scatter"),
    ("biểu đồ đặc trưng", "draw_feature_scatter"),
    ("scatter đặc trưng", "draw_feature_scatter"),
    ("draw feature", "draw_feature_scatter"),
    ("vẽ feature", "draw_feature_scatter"),

    # ── draw_pie_chart ────────────────────────────────────────
    ("vẽ biểu đồ tròn", "draw_pie_chart"),
    ("biểu đồ tròn", "draw_pie_chart"),
    ("draw pie chart", "draw_pie_chart"),
    ("pie chart", "draw_pie_chart"),
    ("vẽ cho tôi cái biểu đồ tròn", "draw_pie_chart"),
    ("vẽ chart tròn", "draw_pie_chart"),
    ("thống kê hình tròn", "draw_pie_chart"),
    ("hãy vẽ biểu đồ tròn", "draw_pie_chart"),
    ("vẽ giúp tôi biểu đồ tròn", "draw_pie_chart"),
    ("tạo biểu đồ tròn", "draw_pie_chart"),
    ("tạo cho tôi biểu đồ tròn", "draw_pie_chart"),
    ("cho tôi biểu đồ tròn", "draw_pie_chart"),
    ("vẽ pie chart", "draw_pie_chart"),

    # ── predict_patient ───────────────────────────────────────
    ("dự đoán bệnh nhân", "predict_patient"),
    ("predict patient", "predict_patient"),
    ("dự đoán ca bệnh mới", "predict_patient"),
    ("nhập bệnh nhân mới", "predict_patient"),
    ("mở form dự đoán", "predict_patient"),
    ("dự đoán giai đoạn bệnh", "predict_patient"),
    ("chẩn đoán bệnh nhân", "predict_patient"),

    # ── train_model ───────────────────────────────────────────
    ("huấn luyện mô hình", "train_model"),
    ("train model", "train_model"),
    ("dự đoán", "train_model"),
    ("chạy model", "train_model"),
    ("phân loại", "train_model"),
    ("predict", "train_model"),
    ("huấn luyện", "train_model"),
    ("machine learning", "train_model"),
]

# Intent → Function name mapping (for Practice.py integration)
INTENT_TO_FUNCTION = {
    "load_data": "PrintData_Clicked",
    "remove_columns": "PrintRemoveCo_Clicked",
    "replace_row": "PrintChangeRow_Clicked",
    "remove_row": "PrintRemoveRow_Clicked",
    "describe": "PrintDescribe",
    "check_null": "PrintChecknull_Clicked",
    "find_popular": "PrintPopularValue_Clicked",
    "reload": "ReLoad_Clicked",
    "remove_isolated": "printIsolate_Clicked",
    "clear": "ClearTextBox_Clicked",
    "draw_line_chart": "drawchartLine_Clicked",
    "print_zscore": "PrintZscore_Clicked",
    "draw_scatter": "drawchartsca_Clicked",
    "minmax_scaler": "PrintMinMaxScaler_Clicked",
    "filter_kbest": "ChoosefiltColumnsKbest_Clicked",
    "filter_pca": "ChoosefiltColumnsPCA_Clicked",
    "show_features": "ShowColumnsFeature_Clicked",
    "delete_chart": "delete_chart_Clicked",
    "draw_feature_scatter": "drawcharattributeScatter_Clicked",
    "draw_pie_chart": "draw_pie_chart_Clicked",
    "predict_patient": "OpenPredictDialog_Clicked",
    "train_model": "TrainModel_Clicked",
}

# Readable intent descriptions (Vietnamese)
INTENT_DESCRIPTIONS = {
    "load_data": "📊 Hiển thị dữ liệu",
    "remove_columns": "🗑️ Xóa cột không cần thiết",
    "replace_row": "🔄 Thay thế giá trị thiếu",
    "remove_row": "❌ Xóa hàng có giá trị null",
    "describe": "📋 Mô tả thống kê dữ liệu",
    "check_null": "🔍 Kiểm tra giá trị null",
    "find_popular": "📈 Tìm giá trị phổ biến",
    "reload": "🔄 Tải lại dữ liệu",
    "remove_isolated": "🎯 Loại bỏ giá trị ngoại lai",
    "clear": "🧹 Xóa nội dung màn hình",
    "draw_line_chart": "📉 Vẽ biểu đồ đường",
    "print_zscore": "📊 Xuất ma trận Z-Score",
    "draw_scatter": "📈 Vẽ biểu đồ scatter",
    "minmax_scaler": "⚖️ Chuẩn hóa MinMax",
    "filter_kbest": "🔬 Trích lọc K-Best",
    "filter_pca": "🔬 Trích lọc PCA",
    "show_features": "📋 Hiển thị kết quả trích lọc",
    "delete_chart": "🗑️ Xóa biểu đồ",
    "draw_feature_scatter": "📈 Vẽ biểu đồ đặc trưng",
    "draw_pie_chart": "🥧 Vẽ biểu đồ tròn",
    "predict_patient": "🩺 Dự đoán ca bệnh mới",
    "train_model": "🤖 Huấn luyện mô hình ML",
}


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Mô hình NLP phân loại Intent từ câu lệnh tiếng Việt.

    Pipeline:
        1. TfidfVectorizer: Tokenize + tính TF-IDF
        2. LogisticRegression: Phân loại intent

    Ưu điểm so với if-else:
        - Hiểu được các biến thể câu lệnh (typo-tolerant)
        - Có confidence score (biết khi nào không chắc)
        - Dễ mở rộng (chỉ cần thêm training data)
    """

    def __init__(self, confidence_threshold: float = 0.12):
        """
        Args:
            confidence_threshold: Ngưỡng tin cậy tối thiểu để chấp nhận intent.
                                  Nếu confidence < threshold → trả về "unknown".
        """
        self.confidence_threshold = confidence_threshold
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=5000,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=5.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ])
        self.is_trained = False
        self.intent_labels = []

    def train(self, training_data: list = None) -> dict:
        """
        Huấn luyện mô hình từ training data.

        Args:
            training_data: List of (text, intent) tuples. Mặc định dùng TRAINING_DATA.

        Returns:
            dict: Thông tin huấn luyện (accuracy, n_intents, n_samples)
        """
        if training_data is None:
            training_data = TRAINING_DATA

        texts = [t[0] for t in training_data]
        labels = [t[1] for t in training_data]
        self.intent_labels = sorted(set(labels))

        print("=" * 50)
        print("🧠 HUẤN LUYỆN MÔ HÌNH INTENT CLASSIFICATION")
        print("=" * 50)
        print(f"  📊 Số lượng mẫu: {len(texts)}")
        print(f"  🏷️  Số lượng intent: {len(self.intent_labels)}")
        print(f"  📝 Intents: {', '.join(self.intent_labels)}")

        # Cross-validation
        cv_scores = cross_val_score(self.pipeline, texts, labels, cv=3, scoring="accuracy")
        print(f"\n  📈 Cross-validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit trên toàn bộ data
        self.pipeline.fit(texts, labels)
        self.is_trained = True

        train_acc = self.pipeline.score(texts, labels)
        print(f"  ✅ Training Accuracy: {train_acc:.4f}")

        info = {
            "n_samples": len(texts),
            "n_intents": len(self.intent_labels),
            "cv_accuracy": cv_scores.mean(),
            "train_accuracy": train_acc,
        }

        print(f"\n  ✅ Mô hình đã sẵn sàng!")
        return info

    def predict(self, text: str) -> dict:
        """
        Dự đoán intent từ câu lệnh.

        Args:
            text: Câu lệnh đầu vào (tiếng Việt)

        Returns:
            dict: {
                "intent": str,           # Tên intent
                "confidence": float,      # Độ tin cậy (0-1)
                "function": str,          # Tên hàm tương ứng
                "description": str,       # Mô tả intent
                "is_confident": bool,     # True nếu confidence >= threshold
            }
        """
        if not self.is_trained:
            self.train()

        text_lower = text.lower().strip()
        proba = self.pipeline.predict_proba([text_lower])[0]
        max_idx = proba.argmax()
        intent = self.pipeline.classes_[max_idx]
        confidence = proba[max_idx]
        is_confident = confidence >= self.confidence_threshold

        return {
            "intent": intent if is_confident else "unknown",
            "confidence": confidence,
            "function": INTENT_TO_FUNCTION.get(intent, None),
            "description": INTENT_DESCRIPTIONS.get(intent, "❓ Không xác định"),
            "is_confident": is_confident,
            "all_probabilities": {
                self.pipeline.classes_[i]: float(proba[i])
                for i in range(len(proba))
            },
        }

    def predict_top_k(self, text: str, k: int = 3) -> list:
        """
        Trả về top-k intent có xác suất cao nhất.

        Args:
            text: Câu lệnh đầu vào
            k:    Số lượng intent trả về

        Returns:
            list of dict: Top-k intents với confidence
        """
        if not self.is_trained:
            self.train()

        text_lower = text.lower().strip()
        proba = self.pipeline.predict_proba([text_lower])[0]
        top_indices = proba.argsort()[::-1][:k]

        results = []
        for idx in top_indices:
            intent = self.pipeline.classes_[idx]
            results.append({
                "intent": intent,
                "confidence": float(proba[idx]),
                "description": INTENT_DESCRIPTIONS.get(intent, ""),
            })

        return results

    def save_model(self, filepath: str = None):
        """Lưu mô hình đã huấn luyện ra file."""
        filepath = filepath or str(MODEL_FILE)
        with open(filepath, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"💾 Mô hình đã lưu: {filepath}")

    def load_model(self, filepath: str = None):
        """Tải mô hình từ file."""
        filepath = filepath or str(MODEL_FILE)
        with open(filepath, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True
        print(f"📂 Mô hình đã tải: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════════════

def demo():
    """Chạy demo phân loại intent."""
    classifier = IntentClassifier()
    classifier.train()

    test_phrases = [
        "Vẽ cho tôi cái biểu đồ đường",
        "xem dữ liệu đi",
        "kiểm tra xem có null không",
        "chuẩn hóa dữ liệu cho tôi",
        "xóa biểu đồ đi",
        "trích lọc pca",
        "huấn luyện model đi",
        "tìm giá trị phổ biến",
        "loại bỏ outlier",
        "xin chào",  # unknown intent
        "vẽ biểu đồ tròn",  # close to draw_scatter
        "giúp tôi xóa cột",
        "chạy machine learning",
    ]

    print(f"\n{'=' * 60}")
    print("🧪 DEMO: PHÂN LOẠI INTENT")
    print(f"{'=' * 60}")

    for phrase in test_phrases:
        result = classifier.predict(phrase)
        status = "✅" if result["is_confident"] else "❌"
        print(f"\n  Input:  \"{phrase}\"")
        print(f"  {status} Intent: {result['intent']} ({result['confidence']:.2%})")
        print(f"     → {result['description']}")

        # Top 3
        top3 = classifier.predict_top_k(phrase, k=3)
        for i, t in enumerate(top3):
            print(f"     [{i+1}] {t['intent']}: {t['confidence']:.2%}")


if __name__ == "__main__":
    demo()
