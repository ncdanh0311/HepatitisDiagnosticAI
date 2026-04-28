from tkinter import messagebox

import matplotlib.pyplot as plt

import Process as process


class Chart(process.Process):
    def __init__(self, text, lbl, lbox, tree, cbbluachon, ax1, canvas1):
        super().__init__(text, lbl, lbox, tree, cbbluachon)
        self.ax1 = ax1
        self.canvas1 = canvas1

    def drawchartsca_Clicked(self):
        try:
            cols = self.zscore
            zscore_data = self.zscore
            for col in cols:
                data = zscore_data[col].reset_index()
                self.ax1.scatter(data["index"], data[col], label=col)
            self.ax1.legend()
            self.ax1.set_title("Biểu đồ Z-Score Scatter", color="red", fontsize=11)
            self.ax1.set_xlabel("Số dòng", color="blue", fontsize=9)
            self.ax1.set_ylabel("Giá trị Zscore", color="black", fontsize=9)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Vui lòng thực hiện từng bước")

    def drawchartLine_Clicked(self):
        try:
            zscore_data = self.zscore
            zscore_data.plot(kind="line", legend=True, ax=self.ax1, fontsize=9)
            self.ax1.set_title("Biểu đồ Z-Score Line", color="red", fontsize=11)
            self.ax1.set_xlabel("Số dòng", color="blue", fontsize=9)
            self.ax1.set_ylabel("Giá trị Zscore", color="black", fontsize=9)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Vui lòng thực hiện từng bước")

    def drawcharattributeScatter_Clicked(self):
        try:
            x_values = self.temp.reset_index()
            y_values = self.temp[self.listcolumns]
            self.ax1.scatter(x_values["index"], y_values, label=list(self.listcolumns))
            self.ax1.legend()
            self.ax1.set_title("Biểu đồ Scatter thuộc tính", color="red", fontsize=11)
            self.ax1.set_xlabel("Số dòng", color="blue", fontsize=9)
            self.ax1.set_ylabel("Giá trị", color="black", fontsize=9)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Bạn chưa chọn cột để vẽ hoặc cần chọn từng cột để vẽ")

    def drawcharattributePlot_Clicked(self):
        try:
            sorted_df = self.temp.sort_values(by=self.listcolumns)
            x_values = sorted_df[self.listcolumns].values.flatten()
            y_values = sorted_df[self.listcolumns].diff().values.flatten()
            self.ax1.plot(x_values, y_values, linewidth=2)
            self.ax1.set_xlabel("Thuộc tính đặc trưng")
            self.ax1.set_ylabel("Độ biến thiên")
            self.ax1.set_title("Biểu đồ Plot biểu diễn cột đặc trưng")
            self.ax1.legend(list([self.listcolumns]), loc="upper left")
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Bạn chưa chọn cột để vẽ hoặc cần chọn từng cột để vẽ")

    def drawcharattributeBar_Clicked(self):
        try:
            x_values = self.temp.reset_index()
            y_values = self.temp[self.listcolumns].values.flatten()
            self.ax1.bar(x_values["index"], y_values)
            self.ax1.set_title("Biểu đồ cột biểu diễn cột đặc trưng", color="red", fontsize=11)
            self.ax1.set_xlabel("Số dòng", color="blue", fontsize=9)
            self.ax1.set_ylabel("Giá trị", color="black", fontsize=9)
            self.ax1.legend(list([self.listcolumns]), loc="upper left")
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Bạn chưa chọn cột để vẽ hoặc cần chọn từng cột để vẽ")

    def draw_heatmap_Clicked(self):
        try:
            corr_matrix = self.temp.corr()
            self.ax1.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    self.ax1.text(
                        j,
                        i,
                        "{:.2f}".format(corr_matrix.iloc[i, j]),
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )
            self.ax1.set_xticks(range(corr_matrix.shape[0]))
            self.ax1.set_yticks(range(corr_matrix.shape[1]))
            self.ax1.set_xticklabels(corr_matrix.columns)
            self.ax1.set_yticklabels(corr_matrix.columns)
            plt.setp(self.ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            self.ax1.set_title("Biểu đồ Heatmap Z-Score", color="red", fontsize=11)
            self.ax1.set_xlabel("Các biến", color="blue", fontsize=9)
            self.ax1.set_ylabel("Các biến", color="black", fontsize=9)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thông báo", "Vui lòng thực hiện từng bước")

    def draw_pie_chart_Clicked(self):
        try:
            category_counts = self.data["Category"].value_counts().sort_index()
            labels = [f"Category {idx}" for idx in category_counts.index]
            self.ax1.pie(
                category_counts.values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
            )
            self.ax1.set_title("Bieu do tron phan bo Category", color="red", fontsize=11)
            self.canvas1.draw()
            self.canvas1.get_tk_widget().place(x=290, y=0)
        except Exception:
            messagebox.showerror("Thong bao", "Vui long tai du lieu truoc khi ve bieu do tron")

    def delete_chart_Clicked(self):
        self.canvas1.get_tk_widget().place_forget()
        self.ax1.clear()
