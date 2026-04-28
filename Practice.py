"""
Module Điều phối Lệnh (Command Controller)
==========================================
Sử dụng NLP Intent Classification thay vì if-else cứng nhắc.

Pipeline:
    Voice/Text Input → NLP IntentClassifier → Intent → Execute Function

"""

from tkinter import END, Button, Entry, Frame, Label, Toplevel, messagebox

import pandas as pd

import Mono as mono
import Process_Chart as process_chart
from nlp_intent import IntentClassifier, INTENT_TO_FUNCTION


class Practice(process_chart.Chart):
    """
    Lớp điều phối lệnh từ người dùng.
    Sử dụng mô hình NLP để hiểu intent thay vì so khớp từ khóa.
    """

    def __init__(self, text, lbl, lbox, tree, cbbluachon, ax1, canvas1, textinput, treesel):
        super().__init__(text, lbl, lbox, tree, cbbluachon, ax1, canvas1)
        self.textinput = textinput
        self.treesel = treesel
        self.prediction_trainer = None

        # Khởi tạo NLP Intent Classifier
        self._init_nlp()

    def _init_nlp(self):
        """Huấn luyện mô hình NLP khi khởi động."""
        try:
            self.classifier = IntentClassifier(confidence_threshold=0.12)
            self.classifier.train()
            self._nlp_ready = True
            print("✅ NLP Intent Classifier đã sẵn sàng!")
        except Exception as e:
            print(f"⚠️ Không thể khởi tạo NLP: {e}")
            print("   Sẽ sử dụng fallback if-else.")
            self._nlp_ready = False

    def _get_intent_function_map(self) -> dict:
        """
        Mapping từ intent label → bound method.
        Bao gồm cả chức năng huấn luyện model mới.
        """
        return {
            "load_data": self.PrintData_Clicked,
            "remove_columns": self.PrintRemoveCo_Clicked,
            "replace_row": self.PrintChangeRow_Clicked,
            "remove_row": self.PrintRemoveRow_Clicked,
            "describe": self.PrintDescribe,
            "check_null": self.PrintChecknull_Clicked,
            "find_popular": self.PrintPopularValue_Clicked,
            "reload": self.ReLoad_Clicked,
            "remove_isolated": self.printIsolate_Clicked,
            "clear": self.ClearTextBox_Clicked,
            "draw_line_chart": self.drawchartLine_Clicked,
            "print_zscore": self.PrintZscore_Clicked,
            "draw_scatter": self.drawchartsca_Clicked,
            "minmax_scaler": self.PrintMinMaxScaler_Clicked,
            "filter_kbest": self.ChoosefiltColumnsKbest_Clicked,
            "filter_pca": self.ChoosefiltColumnsPCA_Clicked,
            "show_features": self.ShowColumnsFeature_Clicked,
            "delete_chart": self.delete_chart_Clicked,
            "draw_feature_scatter": self.drawcharattributeScatter_Clicked,
            "draw_pie_chart": self.draw_pie_chart_Clicked,
            "predict_patient": self.OpenPredictDialog_Clicked,
            "train_model": self.TrainModel_Clicked,
        }

    def TrainModel_Clicked(self):
        """Chạy pipeline huấn luyện mô hình ML."""
        try:
            from trainer import run_training_pipeline
            self.text.insert(1.0, "🤖 Đang huấn luyện mô hình ML...\n")
            self.text.insert(END, "Vui lòng đợi trong giây lát...\n\n")
            trainer = run_training_pipeline(binary=False, save_plot=True)
            self.prediction_trainer = trainer
            self.text.insert(END, "\n✅ Huấn luyện hoàn tất!\n")
            self.text.insert(END, f"🏆 Mô hình tốt nhất: {trainer.best_model_name}\n")
            best = trainer.results[trainer.best_model_name]
            self.text.insert(END, f"📊 Accuracy: {best['test_accuracy']:.4f}\n")
            self.text.insert(END, "🧬 Bản huấn luyện: phân loại 5 giai đoạn Category\n")
            self.text.insert(END, f"\n📁 Biểu đồ đã lưu: model_comparison.png\n")
        except Exception as e:
            self.text.insert(1.0, f"❌ Lỗi huấn luyện: {str(e)}\n")

    def _ensure_prediction_trainer(self):
        """Load or train the multi-class model used for patient-level prediction."""
        if self.prediction_trainer is None:
            from trainer import run_training_pipeline

            self.Showtext("🤖 Chưa có model suy luận. Đang tự động huấn luyện model Category...")
            self.prediction_trainer = run_training_pipeline(binary=False, save_plot=True)
        return self.prediction_trainer

    def OpenPredictDialog_Clicked(self):
        """Open a patient form and predict Category for a new case."""
        dialog = Toplevel()
        dialog.title("Predict New Patient")
        dialog.geometry("430x500")
        dialog.resizable(False, False)
        dialog.configure(bg="#F8FBFF")

        Label(
            dialog,
            text="Patient Category Prediction",
            font=("Arial", 14, "bold"),
            bg="#F8FBFF",
            fg="#0B5394",
        ).pack(pady=12)

        Label(
            dialog,
            text="Nhập đầy đủ chỉ số sinh hóa của bệnh nhân mới",
            font=("Arial", 9),
            bg="#F8FBFF",
            fg="#333333",
        ).pack(pady=2)

        fields = [
            ("Age", "Tuổi"),
            ("Sex", "Giới tính (m/f)"),
            ("ALB", "Albumin"),
            ("ALP", "Alkaline Phosphatase"),
            ("ALT", "Alanine Aminotransferase"),
            ("AST", "Aspartate Aminotransferase"),
            ("BIL", "Bilirubin"),
            ("CHE", "Cholinesterase"),
            ("CHOL", "Cholesterol"),
            ("CREA", "Creatinine"),
            ("GGT", "Gamma-Glutamyl Transferase"),
            ("PROT", "Total Proteins"),
        ]

        entries = {}
        for field_name, label_text in fields:
            row = Frame(dialog, bg="#F8FBFF")
            row.pack(fill="x", padx=18, pady=4)
            Label(
                row,
                text=label_text,
                width=22,
                anchor="w",
                font=("Arial", 9, "bold"),
                bg="#F8FBFF",
            ).pack(side="left")
            entry = Entry(row, width=22, font=("Arial", 9))
            entry.pack(side="right")
            entries[field_name] = entry

        Button(
            dialog,
            text="Predict",
            width=16,
            height=2,
            bg="#0072C6",
            fg="#FFFFFF",
            font=("Arial", 10, "bold"),
            command=lambda: self.PredictPatient_Clicked(entries, dialog),
        ).pack(pady=16)

    def PredictPatient_Clicked(self, entries, dialog=None):
        """Predict the disease stage for a newly entered patient."""
        try:
            trainer = self._ensure_prediction_trainer()
            feature_order = list(trainer.X.columns)
            patient_data = {}

            for feature_name in feature_order:
                raw_value = entries[feature_name].get().strip()
                if not raw_value:
                    raise ValueError(f"Thiếu giá trị cho {feature_name}")

                if feature_name == "Sex":
                    sex_value = raw_value.lower()
                    if sex_value not in {"m", "f", "0", "1"}:
                        raise ValueError("Giới tính phải là m, f, 0 hoặc 1")
                    patient_data[feature_name] = 0 if sex_value in {"m", "0"} else 1
                else:
                    patient_data[feature_name] = float(raw_value)

            patient_df = pd.DataFrame([patient_data], columns=feature_order)
            model = trainer.best_model
            pred_code = int(model.predict(patient_df)[0])
            pred_name = trainer.target_names[pred_code]

            confidence_text = ""
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(patient_df)[0][pred_code])
                confidence_text = f" (confidence: {proba:.1%})"

            if pred_name in {"Blood Donor", "Suspect Blood Donor"}:
                summary = f"Khả năng cao là {pred_name}{confidence_text}."
            else:
                summary = f"Cảnh báo {pred_name}{confidence_text}."

            self.Showtext("🩺 Đây là kết quả dự đoán cho ca bệnh mới:")
            self.Showtext(summary)
            mono.Mono_Speak(summary)
            messagebox.showinfo("Prediction Result", summary)

            if dialog is not None:
                dialog.destroy()
        except Exception as exc:
            messagebox.showerror("Prediction Error", str(exc))

    def PrintResult_Clicked(self):
        """
        Xử lý lệnh từ người dùng bằng NLP Intent Classification.

        Flow:
            1. Nhận input từ voice/text
            2. NLP predict intent + confidence
            3. Nếu confident → thực thi hàm tương ứng
            4. Nếu không → hiển thị gợi ý top-3 intent
        """
        command_text = mono.Command().upper().strip()

        if self._nlp_ready:
            self._process_with_nlp(command_text)
        else:
            self._process_with_fallback(command_text)

    def _process_with_nlp(self, command_text: str):
        """Xử lý lệnh bằng NLP Intent Classification."""
        result = self.classifier.predict(command_text)
        intent_map = self._get_intent_function_map()

        if result["is_confident"] and result["intent"] in intent_map:
            # ── Thực thi lệnh ────────────────────────────────
            intent = result["intent"]
            confidence = result["confidence"]

            self.Showtext(f"\n🗣️ {command_text}")
            self.Showtext(f"🧠 Intent: {result['description']} ({confidence:.0%})")

            func = intent_map[intent]
            func()
            self.CompleteCommand()

        else:
            # ── Không hiểu → gợi ý ──────────────────────────
            self.Showtext(f"\n🗣️ {command_text}")
            self.Showtext(f"❓ Không nhận diện được lệnh (confidence: {result['confidence']:.0%})")

            # Gợi ý top-3 intent
            top3 = self.classifier.predict_top_k(command_text, k=3)
            self.Showtext("💡 Có phải bạn muốn:")
            for i, t in enumerate(top3):
                self.Showtext(f"   [{i+1}] {t['description']} ({t['confidence']:.0%})")

            message = mono.Mono_Speak("Bạn nói gì tôi không hiểu rõ, bạn hãy nói lại đi!")
            self.Showtext(message)

    def _process_with_fallback(self, command_text: str):
        """
        Fallback: dùng keyword matching nếu NLP không khả dụng.
        (Giữ lại tương thích ngược)
        """
        commands = {
            "DỮ LIỆU": self.PrintData_Clicked,
            "CỘT": self.PrintRemoveCo_Clicked,
            "THAY THẾ HÀNG": self.PrintChangeRow_Clicked,
            "HÀNG": self.PrintRemoveRow_Clicked,
            "LẤY MÔ TẢ": self.PrintDescribe,
            "KIỂM TRA RỖNG": self.PrintChecknull_Clicked,
            "TÌM GIÁ TRỊ PHỔ BIẾN": self.PrintPopularValue_Clicked,
            "TẠO MỚI": self.ReLoad_Clicked,
            "GIÁ TRỊ CÁ BIỆT": self.printIsolate_Clicked,
            "NỘI DUNG": self.ClearTextBox_Clicked,
            "VẼ BIỂU ĐỒ ĐƯỜNG": self.drawchartLine_Clicked,
            "XUẤT MA TRẬN": self.PrintZscore_Clicked,
            "VẼ BIỂU ĐỒ KHÁC": self.drawchartsca_Clicked,
            "CHUẨN HÓA RỜI RẠC": self.PrintMinMaxScaler_Clicked,
            "TRÍCH LỌC K": self.ChoosefiltColumnsKbest_Clicked,
            "TRÍCH LỌC P": self.ChoosefiltColumnsPCA_Clicked,
            "HIỂN THỊ KẾT QUẢ": self.ShowColumnsFeature_Clicked,
            "XÓA BIỂU ĐỒ": self.delete_chart_Clicked,
            "VẼ CỘT ĐẶC TRƯNG": self.drawcharattributeScatter_Clicked,
            "BIỂU ĐỒ TRÒN": self.draw_pie_chart_Clicked,
            "DỰ ĐOÁN": self.OpenPredictDialog_Clicked,
            "BỆNH NHÂN": self.OpenPredictDialog_Clicked,
            "HUẤN LUYỆN": self.TrainModel_Clicked,
        }

        for key, command_func in commands.items():
            if key in command_text:
                command_func()
                self.Showtext("\n" + command_text)
                self.CompleteCommand()
                break
        else:
            message = mono.Mono_Speak("Bạn nói gì tôi không hiểu, bạn hãy nói lại đi!")
            self.Showtext("\n" + command_text + "\n" + message)

    def CompleteCommand(self):
        """Phát âm thanh hoàn thành."""
        mono.Mono_Speak("Tôi đã hoàn thành yêu cầu.")

    def ClearTextBox_Clicked(self):
        """Xóa toàn bộ nội dung hiển thị."""
        self.text.delete(1.0, END)
        self.textinput.delete(1.0, END)
        self.RemoveTree_Clicked(self.tree)
        self.delete_chart_Clicked()

    def Showtext(self, data):
        """Hiển thị text lên vùng input."""
        self.textinput.insert(END, data)
        self.textinput.insert(END, "\n")

    def SelectValueTree_Clicked(self, event):
        """Xử lý sự kiện chọn item trên Treeview."""
        self.RemoveTree_Clicked(self.treesel)
        self.textinput.delete(1.0, END)
        selected_items = self.tree.selection()
        if selected_items:
            selected_item = selected_items[0]
            values = self.tree.item(selected_item, "values")
            columns = self.tree["columns"]
            self.treesel["columns"] = columns
            column_names = [self.tree.heading(col)["text"] for col in columns]
            for col in column_names:
                self.treesel.heading(col, text=col)
                self.treesel.column(col, width=63)
            self.textinput.insert(1.0, "Bạn đã chọn:")
            self.treesel.insert("", "end", values=values)
            self.treesel.place(x=284, y=510, height=50)
        else:
            self.textinput.insert(1.0, "Bạn chưa chọn gì")
