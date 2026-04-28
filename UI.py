from pathlib import Path
from tkinter import BOTH, EXTENDED, GROOVE, LEFT, SUNKEN, Button, Frame, Label, Listbox, Menu, Tk
from tkinter import scrolledtext, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import Game as game
import Practice as practice


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"


class HepatitisUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Hepatitis EDA")
        self.root.geometry("350x450")
        self.root.iconbitmap(str(ASSETS_DIR / "app.ico"))
        self.root.resizable(False, False)
        self.root.config(background="#f5f5f5")

        title_label = Label(
            self.root,
            text="Hepatitis EDA",
            font=("Arial", 30, "bold"),
            fg="#0072c6",
            bg="#f5f5f5",
        )
        title_label.pack(pady=30)

        button_frame = Frame(self.root, bg="#f5f5f5")
        button_frame.pack(pady=30)

        start_btn = Button(
            button_frame,
            text="Start EDA",
            command=self.phan_tich,
            height=3,
            width=15,
            font=("Arial", 10),
            fg="#ffffff",
            bg="#0072c6",
            activebackground="#2196f3",
            activeforeground="#ffffff",
            relief=GROOVE,
        )
        start_btn.pack(padx=10, pady=10)

        vis_btn = Button(
            button_frame,
            text="Start Game",
            command=self.Run_Game,
            height=3,
            width=15,
            font=("Arial", 10),
            fg="#ffffff",
            bg="#4caf50",
            activebackground="#2196f3",
            activeforeground="#ffffff",
            relief=GROOVE,
        )
        vis_btn.pack(padx=10, pady=10)

        exit_btn = Button(
            button_frame,
            text="Exit",
            command=self.root.destroy,
            height=3,
            width=15,
            font=("Arial", 10),
            fg="#ffffff",
            bg="#f44336",
            activebackground="#2196f3",
            activeforeground="#ffffff",
            relief=GROOVE,
        )
        exit_btn.pack(padx=10, pady=10)

    def phan_tich(self):
        self.wd = Tk()
        self.wd.geometry("1200x650")
        self.wd.title("Hepatitis EDA")
        self.wd.resizable(False, False)
        self.wd.iconbitmap(str(ASSETS_DIR / "app.ico"))

        self.title = Label(self.wd, text="DATA ANALYSIS HEPATITIS EDA", font=("Arial", 18, "bold"))
        self.title.pack(pady=10)

        self.frame2 = Frame(self.wd, bg="#ECECEC", bd=2)
        self.frame2.pack(pady=7)

        self.huylblinput = Label(self.wd, text="Thuộc tính Input", font=("Arial", 10, "bold"))
        self.huylblinput.place(x=12, y=100)
        self.huylbloutput = Label(self.wd, text="Thuộc tính Output", font=("Arial", 10, "bold"))
        self.huylbloutput.place(x=12, y=68)

        self.listbox1 = Listbox(self.frame2, height=16, width=35, font=("Arial", 9, "bold"), selectmode=EXTENDED, bd=3)
        self.listbox1.pack(side=LEFT, padx=10)
        self.hienthi = Label(self.wd, text="Số lượng:", font=("Arial", 10, "bold"))
        self.hienthi.place(x=12, y=420)
        self.lblSoLuong = Label(
            self.wd,
            relief=SUNKEN,
            borderwidth=3,
            width=13,
            height=1,
            font=("Arial", 10, "bold"),
            fg="#0072c6",
            bg="#ffffff",
        )
        self.lblSoLuong.place(x=100, y=420)

        self.style = ttk.Style(self.wd)
        self.style.theme_use("alt")
        self.style.configure("Treeview", background="white", rowheight=25, foreground="black", font=("Arial", 9))
        self.style.map("Treeview", background=[("selected", "blue")])
        self.style.configure("Treeview.Heading", font=("Arial", 9, "bold"))
        self.tree = ttk.Treeview(self.wd, show="headings")

        self.textinput2 = scrolledtext.ScrolledText(self.frame2, height=25, width=170, bd=3, wrap="none")
        self.textinput2.pack(side=LEFT, padx=7)
        self.textinput1 = scrolledtext.ScrolledText(self.wd, height=7, width=111, bd=3)
        self.textinput1.place(x=279, y=475)
        self.trees = ttk.Treeview(self.wd, show="headings")

        self.figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.textinput2)

        self.cbbluachon = ttk.Combobox(self.wd)
        self.cbbluachon.configure(width=13, height=10, font=("Arial", 9, "bold"))
        self.cbbluachon.place(x=150, y=70)

        self.controller = practice.Practice(
            self.textinput2,
            self.lblSoLuong,
            self.listbox1,
            self.tree,
            self.cbbluachon,
            self.ax,
            self.canvas,
            self.textinput1,
            self.trees,
        )

        self.listbox1.bind("<Button-3>", self.controller.ShowPopupMenu_Clicked)
        self.tree.bind("<<TreeviewSelect>>", self.controller.SelectValueTree_Clicked)

        self.btnrequest = Button(
            self.wd,
            text="Request",
            height=2,
            width=6,
            command=self.controller.PrintResult_Clicked,
            bg="#4CAF50",
            fg="#ffffff",
            font=("Arial", 10),
            activebackground="yellow",
            activeforeground="black",
            relief=GROOVE,
            bd=3,
        )
        self.btnrequest.place(x=10, y=500)

        self.btnpredict = Button(
            self.wd,
            text="Predict",
            height=2,
            width=6,
            command=self.controller.OpenPredictDialog_Clicked,
            bg="#8E44AD",
            fg="#ffffff",
            font=("Arial", 10),
            activebackground="yellow",
            activeforeground="black",
            relief=GROOVE,
            bd=3,
        )
        self.btnpredict.place(x=10, y=550)

        self.btn = Button(
            self.wd,
            text="Clear",
            height=2,
            width=6,
            command=self.controller.ClearTextBox_Clicked,
            font=("Arial", 10),
            activebackground="yellow",
            activeforeground="black",
            relief=GROOVE,
            bd=3,
        )
        self.btn.place(x=210, y=500)

        self.Add = Button(
            self.wd,
            text="Add Cols",
            height=2,
            width=6,
            font=("Arial", 10),
            activebackground="yellow",
            activeforeground="black",
            relief=GROOVE,
            command=self.controller.InsertData_Clicked,
            bd=3,
        )
        self.Add.place(x=110, y=500)

        self.menubar = Menu(self.wd, borderwidth=1, relief="solid")
        self.file_menu = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.file_menu.add_command(label="File New", command=self.controller.ReLoad_Clicked)
        self.file_menu.add_command(label="Load Data", command=self.controller.PrintData_Clicked)
        self.file_menu.add_command(label="Describe", command=self.controller.PrintDescribe)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.wd.destroy)

        self.edit_menu = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.edit_menu.add_command(label="Check line null", command=self.controller.PrintChecknull_Clicked)
        self.edit_menu.add_command(label="Find Common Value", command=self.controller.PrintPopularValue_Clicked)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Delete unimportant columns", command=self.controller.PrintRemoveCo_Clicked)
        self.edit_menu.add_command(label="Replace row with common value", command=self.controller.PrintChangeRow_Clicked)
        self.edit_menu.add_command(label="Delete line with null value", command=self.controller.PrintRemoveRow_Clicked)

        self.matrix_menu = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.matrix_menu.add_command(label="Matrix Zscore", command=self.controller.PrintZscore_Clicked)
        self.matrix_menu.add_separator()
        self.matrix_menu.add_command(label="Draw chart Zscore Scatter", command=self.controller.drawchartsca_Clicked)
        self.matrix_menu.add_command(label="Draw chart Zscore Line", command=self.controller.drawchartLine_Clicked)
        self.matrix_menu.add_command(label="Remove Isolated", command=self.controller.printIsolate_Clicked)

        self.MinMax_s = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.MinMax_s.add_command(label="Print MinMax_Scaler", command=self.controller.PrintMinMaxScaler_Clicked)
        self.MinMax_s.add_separator()
        self.MinMax_s.add_command(label="Draw chart Heatmap", command=self.controller.draw_heatmap_Clicked)

        self.Featured = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.Featured.add_command(label="Choose filter Columns Kbest", command=self.controller.ChoosefiltColumnsKbest_Clicked)
        self.Featured.add_command(label="Choose filter Columns PCA", command=self.controller.ChoosefiltColumnsPCA_Clicked)
        self.Featured.add_command(label="Print model extract", command=self.controller.ShowColumnsFeature_Clicked)
        self.Featured.add_separator()
        self.Featured.add_command(label="Draw Attribute Characteristic Scatter", command=self.controller.drawcharattributeScatter_Clicked)
        self.Featured.add_command(label="Draw Attribute Characteristic Plot", command=self.controller.drawcharattributePlot_Clicked)
        self.Featured.add_command(label="Draw Attribute Characteristic Bar", command=self.controller.drawcharattributeBar_Clicked)

        self.prediction_menu = Menu(self.menubar, tearoff=0, font=("Arial", 9, "bold"))
        self.prediction_menu.add_command(label="Train Category Model", command=self.controller.TrainModel_Clicked)
        self.prediction_menu.add_command(label="Predict New Patient", command=self.controller.OpenPredictDialog_Clicked)

        self.menubar.add_cascade(label="File", menu=self.file_menu, font=("Helvetica", 20))
        self.menubar.add_cascade(label="Edit", menu=self.edit_menu, font=("Helvetica", 20))
        self.menubar.add_cascade(label="Matrix", menu=self.matrix_menu, font=("Helvetica", 20))
        self.menubar.add_cascade(label="MinMax_Scaler", menu=self.MinMax_s, font=("Helvetica", 20))
        self.menubar.add_cascade(label="Characteristic Attribute", menu=self.Featured, font=("Helvetica", 20))
        self.menubar.add_cascade(label="Prediction", menu=self.prediction_menu, font=("Helvetica", 20))

        self.wd.config(background="#E8F8F5")
        self.title.config(fg="#34495E", bg="#E8F8F5")
        self.huylblinput.config(fg="#34495E", bg="#D6EAF8")
        self.huylbloutput.config(fg="#34495E", bg="#D6EAF8")
        self.listbox1.config(fg="#000000", bg="#FFFFFF")
        self.hienthi.config(fg="#34495E", bg="#D6EAF8")
        self.lblSoLuong.config(fg="#34495E", bg="#FFFFFF")
        self.frame2.config(bg="#D6EAF8")
        self.textinput2.config(fg="#000000", bg="#FFFFFF")
        self.textinput1.config(fg="#000000", bg="#FFFFFF")
        self.btnrequest.config(fg="#FFFFFF", bg="#3498DB")
        self.btnpredict.config(fg="#FFFFFF", bg="#8E44AD")
        self.btn.config(fg="#FFFFFF", bg="#E74C3C")
        self.Add.config(fg="#FFFFFF", bg="#2ECC71")
        self.menubar.config(background="#FFFFFF", foreground="#34495E")
        self.file_menu.config(background="#FFFFFF", foreground="#34495E")
        self.edit_menu.config(background="#FFFFFF", foreground="#34495E")
        self.matrix_menu.config(background="#FFFFFF", foreground="#34495E")
        self.MinMax_s.config(background="#FFFFFF", foreground="#34495E")
        self.Featured.config(background="#FFFFFF", foreground="#34495E")
        self.prediction_menu.config(background="#FFFFFF", foreground="#34495E")
        self.wd.config(menu=self.menubar)
        self.wd.mainloop()

    def Run_Game(self):
        bg = game.Background()
        car = game.Car()
        obstacles = game.Obstacles()
        score = game.Score()
        runner = game.Run()
        runner.gameStart(bg)
        while True:
            runner.gamePlay(bg, car, obstacles, score)
            runner.gameOver(bg, car, obstacles, score)
