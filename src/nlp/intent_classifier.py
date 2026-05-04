"""
src/nlp/intent_classifier.py
=============================
TF-IDF + LogisticRegression intent classifier for Vietnamese commands.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── Training corpus ──────────────────────────────────────────────────────────

TRAINING_DATA: list[tuple[str, str]] = [
    # load_data
    ("dữ liệu", "load_data"), ("hiển thị dữ liệu", "load_data"),
    ("xem dữ liệu", "load_data"), ("tải dữ liệu", "load_data"),
    ("đọc dữ liệu", "load_data"), ("cho tôi xem dữ liệu", "load_data"),
    ("in dữ liệu ra", "load_data"), ("show data", "load_data"),
    # remove_columns
    ("xóa cột", "remove_columns"), ("loại bỏ cột", "remove_columns"),
    ("bỏ cột không cần", "remove_columns"), ("loại cột thừa", "remove_columns"),
    ("remove columns", "remove_columns"), ("xóa các cột không quan trọng", "remove_columns"),
    # replace_row
    ("thay thế hàng", "replace_row"), ("thay thế giá trị", "replace_row"),
    ("điền giá trị thiếu", "replace_row"), ("fill missing values", "replace_row"),
    ("thay thế null", "replace_row"), ("thay thế NA", "replace_row"),
    # remove_row
    ("xóa hàng", "remove_row"), ("loại bỏ hàng", "remove_row"),
    ("xóa dòng null", "remove_row"), ("remove rows", "remove_row"),
    ("xóa các dòng trống", "remove_row"), ("loại bỏ dòng rỗng", "remove_row"),
    # describe
    ("lấy mô tả", "describe"), ("mô tả dữ liệu", "describe"),
    ("thống kê mô tả", "describe"), ("describe", "describe"),
    ("tóm tắt dữ liệu", "describe"), ("xem thống kê", "describe"),
    # check_null
    ("kiểm tra rỗng", "check_null"), ("kiểm tra null", "check_null"),
    ("tìm giá trị null", "check_null"), ("check null", "check_null"),
    ("xem giá trị thiếu", "check_null"), ("kiểm tra missing", "check_null"),
    # find_popular
    ("tìm giá trị phổ biến", "find_popular"), ("giá trị phổ biến", "find_popular"),
    ("tìm mode", "find_popular"), ("giá trị thường gặp", "find_popular"),
    # reload
    ("tạo mới", "reload"), ("reset", "reload"), ("làm mới", "reload"),
    ("tải lại", "reload"), ("khởi tạo lại", "reload"), ("bắt đầu lại", "reload"),
    # remove_isolated
    ("giá trị cá biệt", "remove_isolated"), ("loại bỏ outlier", "remove_isolated"),
    ("xóa outlier", "remove_isolated"), ("remove outliers", "remove_isolated"),
    ("xóa giá trị ngoại lai", "remove_isolated"), ("xử lý outlier", "remove_isolated"),
    # clear
    ("xóa nội dung", "clear"), ("xóa màn hình", "clear"),
    ("clear", "clear"), ("làm sạch", "clear"), ("dọn dẹp", "clear"),
    # draw_line_chart
    ("vẽ biểu đồ đường", "draw_line_chart"), ("biểu đồ đường", "draw_line_chart"),
    ("line chart", "draw_line_chart"), ("vẽ đường", "draw_line_chart"),
    ("draw line", "draw_line_chart"), ("biểu đồ line", "draw_line_chart"),
    # print_zscore
    ("xuất ma trận", "print_zscore"), ("ma trận zscore", "print_zscore"),
    ("tính zscore", "print_zscore"), ("z-score", "print_zscore"),
    # draw_scatter
    ("vẽ biểu đồ scatter", "draw_scatter"), ("scatter plot", "draw_scatter"),
    ("vẽ phân tán", "draw_scatter"), ("biểu đồ phân tán", "draw_scatter"),
    # minmax_scaler
    ("chuẩn hóa rời rạc", "minmax_scaler"), ("minmax", "minmax_scaler"),
    ("chuẩn hóa dữ liệu", "minmax_scaler"), ("normalize", "minmax_scaler"),
    ("min max scaler", "minmax_scaler"), ("chuẩn hóa", "minmax_scaler"),
    # filter_kbest
    ("trích lọc k", "filter_kbest"), ("kbest", "filter_kbest"),
    ("select k best", "filter_kbest"), ("lọc kbest", "filter_kbest"),
    # filter_pca
    ("trích lọc pca", "filter_pca"), ("pca", "filter_pca"),
    ("giảm chiều", "filter_pca"), ("principal component", "filter_pca"),
    ("phân tích thành phần chính", "filter_pca"),
    # show_features
    ("hiển thị kết quả", "show_features"), ("xem kết quả", "show_features"),
    ("kết quả trích lọc", "show_features"), ("show results", "show_features"),
    # delete_chart
    ("xóa biểu đồ", "delete_chart"), ("xóa chart", "delete_chart"),
    ("delete chart", "delete_chart"), ("dọn biểu đồ", "delete_chart"),
    # draw_feature_scatter
    ("vẽ cột đặc trưng", "draw_feature_scatter"),
    ("vẽ thuộc tính đặc trưng", "draw_feature_scatter"),
    ("biểu đồ đặc trưng", "draw_feature_scatter"),
    # draw_pie_chart
    ("vẽ biểu đồ tròn", "draw_pie_chart"), ("biểu đồ tròn", "draw_pie_chart"),
    ("draw pie chart", "draw_pie_chart"), ("pie chart", "draw_pie_chart"),
    ("vẽ cho tôi cái biểu đồ tròn", "draw_pie_chart"),
    ("tạo biểu đồ tròn", "draw_pie_chart"), ("vẽ pie chart", "draw_pie_chart"),
    # predict_patient
    ("dự đoán bệnh nhân", "predict_patient"), ("predict patient", "predict_patient"),
    ("dự đoán ca bệnh mới", "predict_patient"), ("mở form dự đoán", "predict_patient"),
    ("chẩn đoán bệnh nhân", "predict_patient"),
    # train_model
    ("huấn luyện mô hình", "train_model"), ("train model", "train_model"),
    ("chạy model", "train_model"), ("machine learning", "train_model"),
    ("huấn luyện", "train_model"), ("phân loại bệnh", "train_model"),
]

INTENT_DESCRIPTIONS: dict[str, str] = {
    "load_data":           "📊 Hiển thị dữ liệu",
    "remove_columns":      "🗑️ Xóa cột không cần thiết",
    "replace_row":         "🔄 Thay thế giá trị thiếu",
    "remove_row":          "❌ Xóa hàng có giá trị null",
    "describe":            "📋 Mô tả thống kê dữ liệu",
    "check_null":          "🔍 Kiểm tra giá trị null",
    "find_popular":        "📈 Tìm giá trị phổ biến",
    "reload":              "🔄 Tải lại dữ liệu",
    "remove_isolated":     "🎯 Loại bỏ outlier",
    "clear":               "🧹 Xóa nội dung",
    "draw_line_chart":     "📉 Vẽ biểu đồ đường",
    "print_zscore":        "📊 Xuất ma trận Z-Score",
    "draw_scatter":        "📈 Vẽ biểu đồ scatter",
    "minmax_scaler":       "⚖️ Chuẩn hóa MinMax",
    "filter_kbest":        "🔬 Trích lọc K-Best",
    "filter_pca":          "🔬 Trích lọc PCA",
    "show_features":       "📋 Hiển thị kết quả trích lọc",
    "delete_chart":        "🗑️ Xóa biểu đồ",
    "draw_feature_scatter":"📈 Vẽ biểu đồ đặc trưng",
    "draw_pie_chart":      "🥧 Vẽ biểu đồ tròn",
    "predict_patient":     "🩺 Dự đoán ca bệnh mới",
    "train_model":         "🤖 Huấn luyện mô hình ML",
}


class IntentClassifier:
    """
    TF-IDF + LogisticRegression intent classifier.

    Usage:
        clf = IntentClassifier()
        clf.train()
        result = clf.predict("vẽ biểu đồ đường")
        # result["intent"], result["confidence"], result["description"]
    """

    def __init__(self, confidence_threshold: float = 0.12) -> None:
        self.threshold = confidence_threshold
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4),
                max_features=5000, sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                C=5.0, max_iter=1000, solver="lbfgs", random_state=42,
            )),
        ])
        self.is_trained = False
        self.classes_: list[str] = []

    def train(self, data: list[tuple[str, str]] | None = None) -> dict:
        data = data or TRAINING_DATA
        texts = [t[0] for t in data]
        labels = [t[1] for t in data]
        self.classes_ = sorted(set(labels))

        cv_scores = cross_val_score(self._pipeline, texts, labels, cv=3)
        self._pipeline.fit(texts, labels)
        self.is_trained = True

        return {
            "n_samples": len(texts),
            "n_intents": len(self.classes_),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "train_acc": float(self._pipeline.score(texts, labels)),
        }

    def predict(self, text: str) -> dict:
        if not self.is_trained:
            self.train()
        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        idx = int(proba.argmax())
        intent = self._pipeline.classes_[idx]
        confidence = float(proba[idx])
        is_confident = confidence >= self.threshold
        return {
            "intent": intent if is_confident else "unknown",
            "confidence": confidence,
            "description": INTENT_DESCRIPTIONS.get(intent, "❓ Không xác định"),
            "is_confident": is_confident,
            "all_proba": {
                cls: float(p)
                for cls, p in zip(self._pipeline.classes_, proba)
            },
        }

    def top_k(self, text: str, k: int = 3) -> list[dict]:
        if not self.is_trained:
            self.train()
        proba = self._pipeline.predict_proba([text.lower().strip()])[0]
        top = proba.argsort()[::-1][:k]
        return [
            {
                "intent": self._pipeline.classes_[i],
                "confidence": float(proba[i]),
                "description": INTENT_DESCRIPTIONS.get(
                    self._pipeline.classes_[i], ""
                ),
            }
            for i in top
        ]
