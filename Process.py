"""
Module Xử lý Dữ liệu Hepatitis (Refactored)
=============================================
Tách riêng logic xử lý dữ liệu (DataProcessor) khỏi logic hiển thị UI (UIController).

Kiến trúc:
    DataProcessor  → Xử lý dữ liệu thuần (không phụ thuộc UI)
    UIController   → Kết nối DataProcessor với Tkinter UI

"""

from pathlib import Path
from tkinter import END, FALSE, Menu, messagebox, simpledialog

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "hepatitis.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA PROCESSOR (Pure Data Logic - No UI Dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

class DataProcessor:
    """
    Lớp xử lý dữ liệu hepatitis thuần túy.
    Không phụ thuộc vào bất kỳ thành phần UI nào.

    Attributes:
        data (pd.DataFrame): Dữ liệu hiện tại đang xử lý
        zscore (pd.DataFrame): Ma trận Z-Score
        feature_list (list): Danh sách các cột đặc trưng đã trích lọc
    """

    COLUMN_ORDER = {
        0: "Category", 1: "ALB", 2: "ALP", 3: "ALT", 4: "AST",
        5: "BIL", 6: "CHE", 7: "CHOL", 8: "CREA", 9: "GGT",
        10: "PROT", 11: "Age", 12: "Sex", 13: "Unnamed: 0",
    }

    CATEGORY_MAP = {
        "0=Blood Donor": 0,
        "0s=suspect Blood Donor": 0,
        "1=Hepatitis": 1,
        "2=Fibrosis": 1,
        "3=Cirrhosis": 1,
    }

    SEX_MAP = {"m": 0, "f": 1}

    def __init__(self, filepath: str = None):
        """
        Args:
            filepath: Đường dẫn tới file CSV. Mặc định: hepatitis.csv
        """
        self.filepath = filepath or str(DATA_FILE)
        self.data = self._read_file()
        self.zscore = None
        self.zscore_value = None
        self.number_columns = None
        self.dropped_columns = []
        self.selected_columns = []
        self.feature_list = []

    def _read_file(self) -> pd.DataFrame:
        """Đọc file CSV và mã hóa các cột phân loại."""
        data = pd.read_csv(self.filepath)
        data["Sex"] = data["Sex"].map(self.SEX_MAP)
        data["Category"] = data["Category"].map(self.CATEGORY_MAP)
        return data

    def reload(self):
        """Reset toàn bộ dữ liệu về trạng thái ban đầu."""
        self.data = self._read_file()
        self.zscore = None
        self.zscore_value = None
        self.number_columns = None
        self.dropped_columns.clear()
        self.selected_columns.clear()
        self.feature_list.clear()

    # ── Kiểm tra & Thống kê ──────────────────────────────────

    def check_null(self) -> pd.Series:
        """Kiểm tra số lượng giá trị null cho mỗi cột."""
        return self.data.isnull().sum()

    def describe(self) -> pd.DataFrame:
        """Trả về thống kê mô tả dữ liệu."""
        return self.data.describe()

    def find_popular_value(self) -> pd.DataFrame:
        """Tìm giá trị phổ biến nhất (mode) cho mỗi cột."""
        mode_dict = {}
        for col in self.data.columns.to_list():
            mode_dict[col] = self.data[col].mode()[0]
        return pd.DataFrame(
            list(mode_dict.items()), columns=["Column", "PopularValue"]
        )

    @property
    def shape(self) -> tuple:
        """Trả về kích thước dữ liệu (rows, cols)."""
        return self.data.shape

    @property
    def columns(self) -> list:
        """Trả về danh sách tên cột."""
        return list(self.data.columns)

    # ── Xử lý Dữ liệu ──────────────────────────────────────

    def fill_missing_with_mode(self) -> pd.DataFrame:
        """Thay thế giá trị null bằng giá trị phổ biến nhất (mode)."""
        self.data = self.data.fillna(self.data.mode().iloc[0])
        return self.data

    def drop_null_rows(self) -> pd.DataFrame:
        """Xóa tất cả các dòng có giá trị null."""
        self.data = self.data.dropna(how="any")
        return self.data

    def drop_columns(self, columns: list) -> pd.DataFrame:
        """
        Xóa các cột chỉ định khỏi dữ liệu.

        Args:
            columns: Danh sách tên cột cần xóa

        Returns:
            DataFrame sau khi xóa cột
        """
        if not columns:
            return self.data
        self.data = self.data.drop(columns=columns, axis=1)
        return self.data

    # ── Chuẩn hóa & Biến đổi ────────────────────────────────

    def minmax_scale(self) -> pd.DataFrame:
        """Chuẩn hóa dữ liệu số về khoảng [0, 1] bằng MinMaxScaler."""
        scaler = preprocessing.MinMaxScaler()
        self.data = self.data.select_dtypes(include=np.number)
        scaler.fit(self.data)
        self.data = pd.DataFrame(
            scaler.transform(self.data),
            index=self.data.index,
            columns=self.data.columns,
        )
        return self.data

    def compute_zscore(self) -> pd.DataFrame:
        """Tính Z-Score cho toàn bộ dữ liệu số."""
        self.zscore = np.abs(stats.zscore(self.data))
        return self.zscore

    def remove_outliers(self, threshold: float) -> pd.DataFrame:
        """
        Loại bỏ các dòng có Z-Score vượt ngưỡng.

        Args:
            threshold: Ngưỡng Z-Score (ví dụ: 3.0)

        Returns:
            DataFrame sau khi loại bỏ outliers
        """
        if self.zscore is None:
            raise ValueError("Cần tính Z-Score trước (compute_zscore)")
        self.zscore_value = threshold
        self.data = (
            self.data[(self.zscore < threshold).all(axis=1)]
            .reset_index()
            .drop(columns=["index"])
        )
        return self.data

    # ── Trích lọc Đặc trưng ─────────────────────────────────

    def extract_features_pca(
        self, input_cols: list, output_col: str, n_components: int
    ) -> dict:
        """
        Trích lọc đặc trưng bằng PCA.

        Args:
            input_cols:   Danh sách cột đầu vào
            output_col:   Tên cột đầu ra (target)
            n_components: Số thành phần chính

        Returns:
            dict: {
                "x_transformed": ndarray,
                "y_data": DataFrame,
                "feature_columns": list,
                "variance_ratio": ndarray,
            }
        """
        x_data = self.data[input_cols]
        y_data = self.data[[output_col]]

        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x_data)

        column_names = x_data.columns
        weight_matrix = pca.components_
        feature_columns = []

        for i in range(n_components):
            component = weight_matrix[i]
            feature_column = column_names[np.argmax(abs(component))]
            feature_columns.append(feature_column)

        self.feature_list.append(feature_columns)

        return {
            "x_transformed": x_pca,
            "y_data": y_data,
            "feature_columns": feature_columns,
            "variance_ratio": pca.explained_variance_ratio_,
        }

    def extract_features_kbest(
        self, input_cols: list, output_col: str, k: int
    ) -> dict:
        """
        Trích lọc đặc trưng bằng SelectKBest (chi2).

        Args:
            input_cols: Danh sách cột đầu vào
            output_col: Tên cột đầu ra (target)
            k:          Số lượng đặc trưng tốt nhất cần chọn

        Returns:
            dict: {
                "x_selected": ndarray,
                "y_data": DataFrame,
                "selected_columns": Index,
            }
        """
        x_data = self.data[input_cols]
        y_data = self.data[[output_col]]

        selector = SelectKBest(chi2, k=k)
        selector.fit(x_data, y_data)
        x_selected = selector.transform(x_data)
        selected_columns = x_data.columns[selector.get_support(indices=True)]

        self.feature_list.append(selected_columns)

        return {
            "x_selected": x_selected,
            "y_data": y_data,
            "selected_columns": selected_columns,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. UI CONTROLLER (Bridges DataProcessor ↔ Tkinter UI)
# ═══════════════════════════════════════════════════════════════════════════════

class UIController:
    """
    Điều khiển tương tác giữa DataProcessor và giao diện Tkinter.
    Lớp này kế thừa các phương thức _Clicked để tương thích ngược với UI hiện tại.

    Attributes:
        processor (DataProcessor): Instance xử lý dữ liệu
        text: Widget hiển thị text kết quả
        lbl:  Label hiển thị số lượng
        lbox: Listbox hiển thị danh sách cột
        tree: Treeview hiển thị bảng dữ liệu
        cbbluachon: Combobox chọn cột output
    """

    def __init__(self, text, lbl, lbox, tree, cbbluachon):
        self.text = text
        self.lbl = lbl
        self.lbox = lbox
        self.tree = tree
        self.cbbluachon = cbbluachon
        self.processor = DataProcessor()

    # ── Convenience properties ───────────────────────────────

    @property
    def temp(self):
        """Truy cập trực tiếp vào dữ liệu (backward compatibility)."""
        return self.processor.data

    @temp.setter
    def temp(self, value):
        self.processor.data = value

    @property
    def zscore(self):
        return self.processor.zscore

    @property
    def listcolumns(self):
        return self.processor.selected_columns

    @listcolumns.setter
    def listcolumns(self, value):
        self.processor.selected_columns = value

    @property
    def listfeatures(self):
        return self.processor.feature_list

    @property
    def column_order(self):
        return DataProcessor.COLUMN_ORDER

    # ── UI Helper Methods ────────────────────────────────────

    def _add_combobox_items(self, items):
        """Cập nhật danh sách items cho Combobox."""
        self.cbbluachon["values"] = list(items)

    def _update_listbox_count(self):
        """Cập nhật label hiển thị số lượng items trong Listbox."""
        self.lbl.configure(text=self.lbox.size())

    def load_data_to_treeview(self, data_frame):
        """Load DataFrame vào Treeview widget."""
        columns = list(data_frame.columns)
        self.tree["columns"] = columns

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=63)

        for index, row in data_frame.iterrows():
            values = [row[col] for col in columns]
            self.tree.insert("", index, values=values)

        self.tree.place(x=286, y=65, height=400)

    def ResultFilter(self, x_data, y_out, columns_extract):
        """Hiển thị kết quả trích lọc đặc trưng."""
        self.text.insert(1.0, "Ma trận đầu vào là:\n")
        self.text.insert(END, x_data)
        self.text.insert(END, "\nVector đầu ra là:\n")
        self.text.insert(END, y_out)
        self.text.insert(END, "\nCột đặc trưng:\n")
        self.text.insert(END, list(columns_extract))

    def ResultChoseAtri(self, atri, y_out):
        """Hiển thị thuộc tính đã chọn."""
        self.text.insert(1.0, "Trích lọc thuộc tính đặc trưng\n")
        self.text.insert(END, "\n")
        self.text.insert(END, atri)
        self.text.insert(END, "\nVector đầu ra là:\n")
        self.text.insert(END, y_out)

    # ── Button Click Handlers ────────────────────────────────

    def PrintData_Clicked(self):
        """Hiển thị toàn bộ dữ liệu lên Treeview."""
        self.load_data_to_treeview(self.processor.data)

    def PrintDescribe(self):
        """Hiển thị thống kê mô tả dữ liệu."""
        self.text.insert(1.0, "Mô tả dữ liệu\n")
        self.text.insert(END, self.processor.describe())

    def PrintChecknull_Clicked(self):
        """Hiển thị kết quả kiểm tra null."""
        self.text.insert(1.0, "Kết quả khi kiểm tra null là:\n")
        self.text.insert(END, self.processor.check_null())

    def PrintPopularValue_Clicked(self):
        """Hiển thị giá trị phổ biến nhất."""
        self.text.insert(1.0, "Giá trị phổ biến của từng cột là:\n")
        self.text.insert(END, self.processor.find_popular_value())

    def ReLoad_Clicked(self):
        """Reset dữ liệu về trạng thái ban đầu."""
        self.processor.reload()
        self.lbox.delete(0, END)
        self._update_listbox_count()

    def PrintRemoveCo_Clicked(self):
        """Xóa các cột đã chọn và hiển thị kết quả."""
        raw_data = self.processor._read_file()
        self.text.insert(1.0, "Số lượng cột và dòng ban đầu là:\n")
        self.text.insert(END, raw_data.shape)
        try:
            dropped = self.processor.dropped_columns.copy()
            if dropped:
                self.processor.drop_columns(dropped)
                self.processor.dropped_columns.clear()
            self._add_combobox_items(self.processor.columns)
            self.text.insert(END, "\n")
            self.text.insert(END, "\nSố lượng cột và dòng sau khi xóa là:\n")
            self.text.insert(END, self.processor.shape)
        except Exception:
            messagebox.showerror("Thông báo", "Cột đó đã xóa rồi")

    def PrintChangeRow_Clicked(self):
        """Thay thế giá trị null bằng mode."""
        result = self.processor.fill_missing_with_mode()
        self.load_data_to_treeview(result)

    def PrintRemoveRow_Clicked(self):
        """Xóa các dòng có giá trị null."""
        raw_data = self.processor._read_file()
        self.text.insert(1.0, "Số lượng dòng và cột ban đầu là:\n")
        self.text.insert(END, raw_data.shape)
        result = self.processor.drop_null_rows()
        self.text.insert(END, "\nSố lượng dòng và cột sau khi xóa là:\n")
        self.text.insert(END, result.shape)
        return result

    def PrintZscore_Clicked(self):
        """Tính và hiển thị Z-Score."""
        zscore_data = self.processor.compute_zscore()
        self.load_data_to_treeview(zscore_data)

    def printIsolate_Clicked(self):
        """Loại bỏ outliers dựa trên Z-Score."""
        try:
            if self.processor.zscore_value is None:
                threshold = simpledialog.askfloat(
                    "Z-score", "Nhập giá trị Z-score:", minvalue=None
                )
                result = self.processor.remove_outliers(threshold)
            else:
                result = self.processor.data
            self.text.insert(1.0, "Đã loại bỏ thành công:\n")
            self.text.insert(END, "Còn lại số lượng dòng và cột là:\n")
            self.text.insert(END, result.shape)
        except Exception:
            messagebox.showerror("Thông báo", "Vui lòng thực hiện từng bước")
            self.processor.zscore_value = None

    def PrintMinMaxScaler_Clicked(self):
        """Chuẩn hóa MinMax và hiển thị."""
        scaled_data = self.processor.minmax_scale()
        self.load_data_to_treeview(scaled_data)

    def InsertData_Clicked(self):
        """Thêm tất cả cột vào Listbox."""
        for col_idx in self.column_order:
            col_name = self.column_order[col_idx]
            if self.lbox.get(0, END).count(col_name) > 0:
                continue
            self.lbox.insert(col_idx, col_name)
            self._update_listbox_count()
        self._add_combobox_items(self.column_order.values())

    def ChoseColumns_Clicked(self):
        """Chọn các cột từ Listbox làm input."""
        selected_items = [self.lbox.get(i) for i in self.lbox.curselection()]
        self.processor.selected_columns.extend(selected_items)
        self.lbox.select_clear(0, END)

    def Delete_Clicked(self):
        """Xóa các cột đã chọn khỏi Listbox."""
        i = 0
        while i < self.lbox.size():
            if self.lbox.select_includes(i) == 1:
                self.processor.dropped_columns.append(self.lbox.get(i))
                self.lbox.delete(i)
                i = -1
            i = i + 1
        self._update_listbox_count()
        self.lbox.select_clear(0, END)

    def ViewSelectedColumns_Clicked(self):
        """Hiển thị danh sách cột đã chọn."""
        self.text.insert(1.0, "Những cột đã chọn:\n")
        self.text.insert(END, self.processor.selected_columns)

    def RemoveSelectedColumns_Clicked(self):
        """Xóa danh sách cột đã chọn."""
        self.text.insert(1.0, "Đã xóa thành công\n")
        self.text.insert(END, self.processor.selected_columns)
        self.processor.selected_columns = []

    def ShowPopupMenu_Clicked(self, e):
        """Hiển thị menu popup chuột phải trên Listbox."""
        if self.lbox.size() > 0:
            pop_menu = Menu(self.lbox, tearoff=FALSE)
            pop_menu.add_command(label="Delete", command=self.Delete_Clicked)
            pop_menu.add_command(label="Chose Columns input", command=self.ChoseColumns_Clicked)
            pop_menu.add_command(label="View Select Columns", command=self.ViewSelectedColumns_Clicked)
            pop_menu.add_command(label="Remove Selected Columns", command=self.RemoveSelectedColumns_Clicked)
            pop_menu.tk_popup(e.x_root, e.y_root)

    def RemoveTree_Clicked(self, tree):
        """Xóa nội dung Treeview."""
        for col in tree["columns"]:
            tree.heading(col, text="")
        for item in tree.get_children():
            tree.delete(item)
        tree.place_forget()

    def ShowColumnsFeature_Clicked(self):
        """Hiển thị kết quả trích lọc đặc trưng."""
        self.text.insert(1.0, "Mô hình trích lọc các thuộc tính đặc trưng\n")
        self.text.insert(END, "\n")
        self.text.insert(END, list(self.processor.feature_list))
        try:
            self.text.insert(END, "\n")
            if len(self.processor.selected_columns) == 0 or not self.cbbluachon.get():
                self.text.insert(
                    END,
                    "Bạn vui lòng quét khối chọn các thuộc tính đặc trưng\n"
                    "đã hiển thị trên màn hình. Sau đó hãy chọn thuộc tính output",
                )
            else:
                self.ResultChoseAtri(
                    self.processor.data[self.processor.selected_columns],
                    self.processor.data[[self.cbbluachon.get()]],
                )
        except Exception:
            messagebox.showerror("Thông báo", "Có lỗi xảy ra")

    def ChoosefiltColumnsKbest_Clicked(self):
        """Trích lọc K-Best và hiển thị."""
        try:
            if self.processor.number_columns is None or len(self.processor.selected_columns) == 0:
                self.processor.number_columns = simpledialog.askinteger(
                    "Số lượng",
                    "Vui lòng nhập số lượng cột đặc trưng:",
                    minvalue=None,
                )
            result = self.processor.extract_features_kbest(
                self.processor.selected_columns,
                self.cbbluachon.get(),
                self.processor.number_columns,
            )
            self.ResultFilter(result["x_selected"], result["y_data"], result["selected_columns"])
        except Exception:
            self.processor.number_columns = None
            messagebox.showerror(
                "Thông báo",
                "Bạn chưa quét khối chọn input hoặc output đầu ra hoặc số lượng cột để trích lọc. Vui lòng thử lại",
            )

    def ChoosefiltColumnsPCA_Clicked(self):
        """Trích lọc PCA và hiển thị."""
        try:
            if self.processor.number_columns is None or len(self.processor.selected_columns) == 0:
                self.processor.number_columns = simpledialog.askinteger(
                    "Số lượng",
                    "Vui lòng nhập số lượng cột đặc trưng:",
                    minvalue=None,
                )
            result = self.processor.extract_features_pca(
                self.processor.selected_columns,
                self.cbbluachon.get(),
                self.processor.number_columns,
            )
            self.ResultFilter(result["x_transformed"], result["y_data"], result["variance_ratio"])
        except Exception:
            self.processor.number_columns = None
            messagebox.showerror(
                "Thông báo",
                "Bạn chưa quét khối chọn input hoặc output đầu ra hoặc số lượng cột để trích lọc. Vui lòng thử lại",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

# Alias để code cũ (Process_Chart.py, Practice.py) vẫn hoạt động
Process = UIController
