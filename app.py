import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib
# Use a non-interactive backend since the plots are generated in worker
# threads and no GUI display is required.
matplotlib.use("Agg")

import os
import glob
from PIL import Image, ImageTk
import threading

from analysis.scatter import ScatterAnalysis
from analysis.histogram import HistogramAnalysis
from analysis.boxplot import BoxAnalysis
from analysis.contour import ContourAnalysis
from analysis.kmeans_analysis import KMeansAnalysis
from analysis.dfa import DFAAnalysis
from analysis.lstm import LSTMAnalysis
from analysis.psa import PSAAnalysis
from analysis.prophet_analysis import ProphetAnalysis
from analysis.mass_estimation import MassEstimation

ctk.set_default_color_theme("blue")
ctk.set_appearance_mode("dark")
class ImageViewer(ctk.CTkFrame):
    """Enhanced Image/Text viewer with navigation, zoom, and pan"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.image_files = []
        self.current_index = 0
        self.orig_image = None
        self.current_display_image = None
        self.tk_image = None
        self.image_id = None
        self.current_zoom = 1.0
        self.setup_ui()

    def setup_ui(self):
        nav_frame = ctk.CTkFrame(self)
        nav_frame.pack(fill="x", pady=(0, 10))

        self.prev_btn = ctk.CTkButton(nav_frame, text="◀ Previous", command=self.prev_image, width=100)
        self.prev_btn.pack(side="left", padx=5)

        self.image_label = ctk.CTkLabel(nav_frame, text="No files")
        self.zoom_label = ctk.CTkLabel(nav_frame, text="Zoom: 100%")
        self.zoom_label.pack(side="right", padx=10)
        self.image_label.pack(side="left", expand=True)

        self.next_btn = ctk.CTkButton(nav_frame, text="Next ▶", command=self.next_image, width=100)
        self.next_btn.pack(side="right", padx=5)

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(fill="both", expand=True)

        self.image_canvas = tk.Canvas(self.image_frame, bg="#1a1a1a", highlightthickness=0, xscrollincrement=1, yscrollincrement=1)
        self.image_canvas.pack(fill="both", expand=True)
        self.image_canvas.bind("<ButtonPress-1>", lambda e: self.image_canvas.scan_mark(e.x, e.y))
        self.image_canvas.bind("<B1-Motion>", lambda e: self.image_canvas.scan_dragto(e.x, e.y, gain=1))
        self.image_canvas.bind("<MouseWheel>", self._zoom_image)
        self.image_canvas.bind("<Button-4>", self._zoom_image)
        self.image_canvas.bind("<Button-5>", self._zoom_image)
        self.image_canvas.bind("<Double-Button-1>", self._reset_zoom)

        self.text_display = ctk.CTkTextbox(self.image_frame, wrap="none")
        self.text_display.configure(state="disabled")

    def _zoom_image(self, event):
        if not self.orig_image:
            return

        factor = 1.1 if (event.delta > 0 or getattr(event, 'num', 0) == 4) else 0.9
        self.current_zoom *= factor

        w, h = max(1, int(self.orig_image.width * self.current_zoom)), max(1, int(self.orig_image.height * self.current_zoom))
        self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.current_display_image)
        self.image_canvas.delete("all")
        self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
        self._update_zoom_label()

    def _reset_zoom(self, event=None):
        if not self.orig_image:
            return
        self.image_canvas.update_idletasks()
        canvas_w = self.image_canvas.winfo_width() or 800
        self.image_canvas.update_idletasks()
        canvas_h = self.image_canvas.winfo_height() or 600
        orig_w, orig_h = self.orig_image.size
        scale_w = canvas_w / orig_w
        scale_h = canvas_h / orig_h
        self.current_zoom = min(scale_w, scale_h, 1.0)
        w, h = max(1, int(orig_w * self.current_zoom)), max(1, int(orig_h * self.current_zoom))
        self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.current_display_image)
        self.image_canvas.delete("all")
        self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

    def _center_x(self, width):
        canvas_width = self.image_canvas.winfo_width() or 800
        return max((canvas_width - width) // 2, 0)

    def _center_y(self, height):
        canvas_height = self.image_canvas.winfo_height() or 600
        return max((canvas_height - height) // 2, 0)

    def load_images(self, image_files):
        self.image_files = [f for f in image_files if os.path.exists(f)]
        self.current_index = 0
        self.update_display()

    def update_display(self):
        self.image_canvas.pack_forget()
        self.text_display.pack_forget()

        if not self.image_files:
            self.image_label.configure(text="No files")
            return

        self.prev_btn.configure(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_index < len(self.image_files) - 1 else "disabled")

        filepath = self.image_files[self.current_index]
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        self.image_label.configure(text=f"{filename} ({self.current_index+1}/{len(self.image_files)})")

        if ext in (".png", ".jpg", ".jpeg", ".gif"):
            self.orig_image = Image.open(filepath)

            # Resize to fit current canvas size
            canvas_w = self.image_canvas.winfo_width() or 800
            canvas_h = self.image_canvas.winfo_height() or 600

            orig_w, orig_h = self.orig_image.size
            scale_w = canvas_w / orig_w
            scale_h = canvas_h / orig_h
            initial_scale = min(scale_w, scale_h, 1.0)

            self.current_zoom = initial_scale
            w, h = max(1, int(orig_w * self.current_zoom)), max(1, int(orig_h * self.current_zoom))
            self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.current_display_image)

            self.image_canvas.pack(fill="both", expand=True)
            self.image_canvas.delete("all")
            self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

        elif ext in (".csv", ".txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    data = f.read()
                self.text_display.configure(state="normal")
                self.text_display.delete("0.0", "end")
                self.text_display.insert("0.0", data)
                self.text_display.configure(state="disabled")
                self.text_display.pack(fill="both", expand=True)
            except Exception as e:
                print(f"Error loading text: {e}")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def _update_zoom_label(self):
        percent = int(self.current_zoom * 100)
        self.zoom_label.configure(text=f"Zoom: {percent}%")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_display()

class HomeImage(ctk.CTkFrame):
    def __init__(self, parent, image_path, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image = Image.open(image_path)
        self.tk_image = None
        self.image_id = None
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        if not self.image:
            return
        canvas_w, canvas_h = event.width, event.height
        # Fill the entire area, allowing distortions
        resized = self.image.resize((max(canvas_w, 1), max(canvas_h, 1)), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

class MethaneAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(
            "Sistema Integral de Detección y Cuantificación de Emisiones de Metano (SIDEC-Metano)"
        )
        self.geometry("1400x900")

        # Set window title bar icon
        try:
            ico_path = os.path.join(os.path.dirname(__file__), "CH4.ico")
            self.wm_iconbitmap(ico_path)
        except Exception as e:
            print(f"Error loading window icon with iconbitmap: {e}")
            try:
                png_path = os.path.join(os.path.dirname(__file__), "CH4.png")
                icon_pil = Image.open(png_path)
                icon_img = ImageTk.PhotoImage(icon_pil.resize((32, 32)))
                self.iconphoto(True, icon_img)
                self._icon_img = icon_img  # keep a reference
            except Exception as e2:
                print(f"Error loading window icon with iconphoto: {e2}")

        # Load theme icons
        try:
            sun_img = Image.open("sun.png")
            moon_img = Image.open("moon.png")
            self.sun_ctk = ctk.CTkImage(light_image=sun_img, dark_image=sun_img, size=(20, 20))
            self.moon_ctk = ctk.CTkImage(light_image=moon_img, dark_image=moon_img, size=(20, 20))
        except Exception as e:
            print(f"Error loading theme icons: {e}")
            self.sun_ctk = None
            self.moon_ctk = None

        self.tab_frames = {}
        self.active_tab = None
        self.data_file = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_panel = ctk.CTkFrame(main_frame, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        # Top bar contains tab buttons on the left and theme toggle on the right
        top_bar = ctk.CTkFrame(right_panel)
        top_bar.pack(fill="x", pady=(0, 10))

        tab_button_frame = ctk.CTkFrame(top_bar)
        tab_button_frame.pack(side="left", anchor="nw")

        button_names = [
            "Home", "Load Data", "Scatter Plot", "Histogram", "Box Plot", "Contour",
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon", "K-means: Lat-Lon-Methane", 
            "K-means: Time-Lat-Lon-Methane", "DFA", "PSA", "LSTM", "Prophet", "Estimate Mass"
        ]

        num_cols = 6
        self.tab_buttons = {}
        for i, name in enumerate(button_names):
            btn = ctk.CTkButton(
                tab_button_frame,
                text=name,
                command=lambda n=name: self.handle_tab_click(n),
                width=150,
                height=30,
            )
            row = i // num_cols
            col = i % num_cols
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.tab_buttons[name] = btn

        # Theme toggle icon on the top right
        self.theme_icon = ctk.CTkLabel(top_bar, text="")
        self.theme_icon.pack(side="right", padx=5, anchor="ne")
        self.theme_icon.bind("<Button-1>", lambda e: self.toggle_theme())
        self.update_theme_icon()

        self.display_area = ctk.CTkFrame(right_panel)
        self.display_area.pack(fill="both", expand=True)

        self.create_tab_contents(button_names)
        self.switch_tab("Home")

        self.add_section(left_panel, "Data", ["Load Data"])
        self.add_section(left_panel, "Visualization", ["Scatter Plot", "Histogram", "Box Plot", "Contour"])
        self.add_section(left_panel, "Clustering", [
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon",
            "K-means: Lat-Lon-Methane", "K-means: Time-Lat-Lon-Methane"
        ])
        self.add_section(left_panel, "Time Series", ["DFA", "PSA", "LSTM", "Prophet"])
        self.add_section(left_panel, "Mass Estimation", ["Estimate Mass"])

    def create_tab_contents(self, button_names):
        for name in button_names:
            frame = ctk.CTkFrame(self.display_area)
            self.tab_frames[name] = frame

            if name == "Home":
                self.create_home_tab(frame)
            elif name == "Load Data":
                self.create_load_data_tab(frame)
            elif name == "Scatter Plot":
                self.create_scatter_plot_tab(frame)
            elif name == "Histogram":
                self.create_histogram_tab(frame)
            elif name == "Box Plot":
                self.create_box_plot_tab(frame)
            elif name == "Contour":
                self.create_contour_tab(frame)
            elif name == "K-means: Lat-Lon":
                self.create_kmeans_tab(frame, mode="lat_lon")
            elif name == "K-means: Time-Lat-Lon":
                self.create_kmeans_tab(frame, mode="time_lat_lon")
            elif name == "K-means: Lat-Lon-Methane":
                self.create_kmeans_tab(frame, mode="lat_lon_methane")
            elif name == "K-means: Time-Lat-Lon-Methane":
                self.create_kmeans_tab(frame, mode="time_lat_lon_methane")
            elif name == "DFA":
                self.create_dfa_tab(frame)
            elif name == "PSA":
                self.create_psa_tab(frame)
            elif name == "LSTM":
                self.create_lstm_tab(frame)
            elif name == "Prophet":
                self.create_prophet_tab(frame)
            elif name == "Estimate Mass":
                self.create_mass_estimation_tab(frame)
            else:
                ctk.CTkLabel(frame, text=f"{name} Tab", font=("Arial", 16)).pack(pady=20)
                ctk.CTkLabel(frame, text="Feature coming soon...").pack()

    def create_home_tab(self, frame):
        try:
            home_tab = HomeImage(frame, image_path="mexico.png")
            home_tab.pack(fill="both", expand=True)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"(Mexico image not found: {e})").pack()

    # 1) In create_load_data_tab: add a CTkTextbox for the summary
    def create_load_data_tab(self, frame):
        ctk.CTkLabel(frame, text="Load Data", font=("Arial", 18)).pack(pady=20)
        load_btn = ctk.CTkButton(frame, text="Select CSV File", 
                                command=self.load_data_file, width=200)
        load_btn.pack(pady=10)
        
        self.data_status_label = ctk.CTkLabel(frame, text="No data loaded")
        self.data_status_label.pack(pady=10)
        
        # Add a textbox to display the summary (scrollable by default)
        self.summary_text = ctk.CTkTextbox(frame, wrap="none")
        self.summary_text.configure(state="disabled")  # start as read-only
        self.summary_text.pack(fill="both", expand=True, padx=20, pady=10)

    # 2) In load_data_file: load CSV without forcing parse, then update summary_text
    def load_data_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Read CSV without parse_dates so we can catch missing columns
                df = pd.read_csv(file_path)
                self.data_file = file_path
                filename = os.path.basename(file_path)
                self.data_status_label.configure(
                    text=f"✅ Loaded: {filename} ({len(df)} rows)"
                )
                messagebox.showinfo("Success", 
                                    f"Data loaded successfully!\n{len(df)} rows loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.data_status_label.configure(text="❌ Failed to load data")
                return

            # Prepare and display summary in the textbox
            self.summary_text.configure(state="normal")
            self.summary_text.delete("0.0", "end")  # clear previous text
            
            # List all column names
            cols = ", ".join(df.columns)
            self.summary_text.insert("0.0", f"Columns: {cols}\n")
            
            # Check for 'measurement_time' column
            if 'measurement_time' in df.columns:
                try:
                    # Attempt to parse measurement_time as datetime
                    df['measurement_time'] = pd.to_datetime(df['measurement_time'])
                    earliest = df['measurement_time'].min()
                    latest = df['measurement_time'].max()
                    # Format the dates for readability
                    earliest_str = earliest.strftime("%Y-%m-%d %H:%M:%S")
                    latest_str = latest.strftime("%Y-%m-%d %H:%M:%S")
                    self.summary_text.insert("end", f"Earliest measurement time: {earliest_str}\n")
                    self.summary_text.insert("end", f"Latest measurement time:   {latest_str}")
                except Exception:
                    # Parsing failed
                    self.summary_text.insert("end", 
                        "Column 'measurement_time' exists but could not be parsed as datetime.")
            else:
                # Column missing
                self.summary_text.insert("end", 
                    "Column 'measurement_time' not found in dataset.")
            
            self.summary_text.configure(state="disabled")


    def create_scatter_plot_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Scatter Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_scatter_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_scatter_analysis, width=120)
        self.run_scatter_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_scatter_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.scatter_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate scatter plots")
        self.scatter_status.pack(pady=5)
        
        self.scatter_image_viewer = ImageViewer(frame)
        self.scatter_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_histogram_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Histogram Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_histogram_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_histogram_analysis, width=120)
        self.run_histogram_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_histogram_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.histogram_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate histograms")
        self.histogram_status.pack(pady=5)
        
        self.histogram_image_viewer = ImageViewer(frame)
        self.histogram_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_box_plot_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Box Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_box_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_box_analysis, width=120)
        self.run_box_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_box_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.box_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate box plots")
        self.box_status.pack(pady=5)
        
        self.box_image_viewer = ImageViewer(frame)
        self.box_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_contour_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Contour Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_contour_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_contour_analysis, width=120)
        self.run_contour_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_contour_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.contour_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate contour plots")
        self.contour_status.pack(pady=5)
        
        self.contour_image_viewer = ImageViewer(frame)
        self.contour_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_kmeans_tab(self, frame, mode):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text=f"K-means Clustering Analysis ({mode})", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        run_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=lambda: self.run_kmeans_analysis(mode), width=120)
        run_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=lambda: self.refresh_kmeans_images(mode), width=120)
        refresh_btn.pack(side="left", padx=5)
        
        status_label = ctk.CTkLabel(frame, text=f"Click 'Run Analysis' to perform K-means clustering ({mode})")
        status_label.pack(pady=5)
        
        image_viewer = ImageViewer(frame)
        image_viewer.pack(fill="both", expand=True, padx=10, pady=10)
        
        setattr(self, f'run_kmeans_{mode}_btn', run_btn)
        setattr(self, f'kmeans_{mode}_status', status_label)
        setattr(self, f'kmeans_{mode}_image_viewer', image_viewer)

    def create_dfa_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(title_frame, text="DFA (Detrended Fluctuation Analysis)", font=("Arial", 18)).pack(side="left", padx=20, pady=10)

        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)

        self.run_dfa_btn = ctk.CTkButton(btn_frame, text="Run Analysis", command=self.run_dfa_analysis, width=120)
        self.run_dfa_btn.pack(side="left", padx=5)

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", command=self.refresh_dfa_images, width=120)
        refresh_btn.pack(side="left", padx=5)

        self.dfa_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate DFA plot")
        self.dfa_status.pack(pady=5)

        self.dfa_image_viewer = ImageViewer(frame)
        self.dfa_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_psa_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(title_frame, text="PSA (Power Spectral Analysis)", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        self.run_psa_btn = ctk.CTkButton(btn_frame, text="Run Analysis", command=self.run_psa_analysis, width=120)
        self.run_psa_btn.pack(side="left", padx=5)
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", command=self.refresh_psa_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        self.psa_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate PSA plots")
        self.psa_status.pack(pady=5)
        self.psa_image_viewer = ImageViewer(frame)
        self.psa_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_lstm_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="LSTM Time Series Prediction", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_lstm_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_lstm_analysis, width=120)
        self.run_lstm_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_lstm_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.lstm_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate LSTM predictions")
        self.lstm_status.pack(pady=5)
        
        self.lstm_image_viewer = ImageViewer(frame)
        self.lstm_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_prophet_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Prophet Time Series Prediction", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_prophet_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_prophet_analysis, width=120)
        self.run_prophet_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_prophet_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.prophet_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate Prophet predictions")
        self.prophet_status.pack(pady=5)
        
        self.prophet_image_viewer = ImageViewer(frame)
        self.prophet_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_mass_estimation_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Methane Mass Estimation", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_mass_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_mass_estimation, width=120)
        self.run_mass_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_mass_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.mass_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to estimate methane mass")
        self.mass_status.pack(pady=5)
        
        self.mass_image_viewer = ImageViewer(frame)
        self.mass_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def update_theme_icon(self):
        if self.sun_ctk and self.moon_ctk:
            current_mode = ctk.get_appearance_mode().lower()
            if current_mode == "dark":
                self.theme_icon.configure(image=self.moon_ctk)
            else:
                self.theme_icon.configure(image=self.sun_ctk)
        else:
            self.theme_icon.configure(text="Toggle Theme")

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode().lower()
        new_mode = "light" if current_mode == "dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        self.update_theme_icon()

    def add_section(self, panel, title, names):
        ctk.CTkLabel(panel, text=title, font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 0))
        for name in names:
            ctk.CTkButton(panel, text=name, command=lambda n=name: self.handle_tab_click(n)).pack(fill="x", padx=10, pady=5)

    def handle_tab_click(self, name):
        tabs_requiring_data = [
            "Scatter Plot", "Histogram", "Box Plot", "Contour", 
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon", 
            "K-means: Lat-Lon-Methane", "K-means: Time-Lat-Lon-Methane",
            "DFA", "PSA", "LSTM", "Prophet", "Estimate Mass"
        ]
        if name in tabs_requiring_data:
            if not self.data_file:
                messagebox.showwarning("Warning", "Please load a data file first!")
                self.switch_tab("Load Data")
                return
            
            self.switch_tab(name)
            
            if name == "Scatter Plot":
                self.refresh_scatter_images()
            elif name == "Histogram":
                self.refresh_histogram_images()
            elif name == "Box Plot":
                self.refresh_box_images()
            elif name == "Contour":
                self.refresh_contour_images()
            elif name == "K-means: Lat-Lon":
                self.refresh_kmeans_images("lat_lon")
            elif name == "K-means: Time-Lat-Lon":
                self.refresh_kmeans_images("time_lat_lon")
            elif name == "K-means: Lat-Lon-Methane":
                self.refresh_kmeans_images("lat_lon_methane")
            elif name == "K-means: Time-Lat-Lon-Methane":
                self.refresh_kmeans_images("time_lat_lon_methane")
            elif name == "DFA":
                self.refresh_dfa_images()
            elif name == "PSA":
                self.refresh_psa_images()
            elif name == "LSTM":
                self.refresh_lstm_images()
            elif name == "Prophet":
                self.refresh_prophet_images()
            elif name == "Estimate Mass":
                self.refresh_mass_images()
        else:
            self.switch_tab(name)

    def switch_tab(self, name):
        if self.active_tab:
            self.tab_frames[self.active_tab].pack_forget()
            # restore default style for previously active button
            self.tab_buttons[self.active_tab].configure(
                fg_color="transparent",
                text_color=("gray10", "gray90"),
            )

        self.tab_frames[name].pack(fill="both", expand=True)
        # make active tab visible in both light and dark mode
        self.tab_buttons[name].configure(
            fg_color=("gray75", "gray25"),
            text_color=("black", "white"),
        )
        self.active_tab = name

    def _start_thread(self, target, *args):
        """Utility to start a daemon thread."""
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True
        thread.start()

    def refresh_scatter_images(self):
        if not hasattr(self, 'scatter_image_viewer'):
            return
            
        scatter_patterns = [
            'scatter_*.png',
            '*scatter*.png'
        ]
        
        image_files = []
        for pattern in scatter_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.scatter_image_viewer.load_images(image_files)
            self.scatter_status.configure(text=f"✅ Found {len(image_files)} scatter plot(s)")
        else:
            self.scatter_status.configure(text="No scatter plot images found")

    def refresh_histogram_images(self):
        if not hasattr(self, 'histogram_image_viewer'):
            return
            
        histogram_patterns = [
            '*histogram*.png',
            'histogram_*.png'
        ]
        
        image_files = []
        for pattern in histogram_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.histogram_image_viewer.load_images(image_files)
            self.histogram_status.configure(text=f"✅ Found {len(image_files)} histogram(s)")
        else:
            self.histogram_status.configure(text="No histogram images found")

    def refresh_box_images(self):
        if not hasattr(self, 'box_image_viewer'):
            return
            
        box_patterns = [
            '*boxplot*.png',
            'boxplot_*.png'
        ]
        
        image_files = []
        for pattern in box_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.box_image_viewer.load_images(image_files)
            self.box_status.configure(text=f"✅ Found {len(image_files)} box plot(s)")
        else:
            self.box_status.configure(text="No box plot images found")

    def refresh_contour_images(self):
        if not hasattr(self, 'contour_image_viewer'):
            return
            
        contour_patterns = [
            '*contour*.png',
            'contorno_*.png'
        ]
        
        image_files = []
        for pattern in contour_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.contour_image_viewer.load_images(image_files)
            self.contour_status.configure(text=f"✅ Found {len(image_files)} contour plot(s)")
        else:
            self.contour_status.configure(text="No contour plot images found")

    def refresh_kmeans_images(self, mode):
        viewer = getattr(self, f'kmeans_{mode}_image_viewer', None)
        if not viewer:
            return
            
        kmeans_patterns = [
            f'*kmeans_{mode}*.png',
            f'*cluster_{mode}*.png',
            f'tula_kmeans_{mode}_metricas_vs_K.png',
            f'tula_kmeans_{mode}_space_scatter_basemap.png',
            f'tula_methane3_hist_{mode}_cluster_*.png',
            f'tula_methane3_boxplot_{mode}_cluster.png',
            f'tula_methane3_box_swarmplot_{mode}_cluster.png',
            f'tula_kmeans_{mode}_time_scatter.png'
        ]
        
        image_files = []
        for pattern in kmeans_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            viewer.load_images(image_files)
            getattr(self, f'kmeans_{mode}_status').configure(text=f"✅ Found {len(image_files)} K-means plot(s)")
        else:
            getattr(self, f'kmeans_{mode}_status').configure(text="No K-means images found")

    def refresh_dfa_images(self):
        image_files = [f for f in glob.glob("DFA*.png") if os.path.exists(f)]
        if image_files:
            self.dfa_image_viewer.load_images(image_files)
            self.dfa_status.configure(text=f"✅ Found {len(image_files)} DFA plot(s)")
        else:
            self.dfa_status.configure(text="No DFA images found")

    def refresh_psa_images(self):
        image_files = [f for f in glob.glob("*psd*.png") + glob.glob("*detrended*.png") + glob.glob("*autocorrelation*.png") if os.path.exists(f)]
        if image_files:
            self.psa_image_viewer.load_images(image_files)
            self.psa_status.configure(text=f"✅ Found {len(image_files)} PSA plot(s)")
        else:
            self.psa_status.configure(text="No PSA images found")

    def refresh_lstm_images(self):
        if not hasattr(self, 'lstm_image_viewer'):
            return
            
        lstm_patterns = [
            'LSTM*.png',
            '*LSTM*.png'
        ]
        
        image_files = []
        for pattern in lstm_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.lstm_image_viewer.load_images(image_files)
            self.lstm_status.configure(text=f"✅ Found {len(image_files)} LSTM plot(s)")
        else:
            self.lstm_status.configure(text="No LSTM images found")

    def refresh_prophet_images(self):
        if not hasattr(self, 'prophet_image_viewer'):
            return
            
        prophet_patterns = [
            'Prophet*.png',
            '*Prophet*.png'
        ]
        
        image_files = []
        for pattern in prophet_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.prophet_image_viewer.load_images(image_files)
            self.prophet_status.configure(text=f"✅ Found {len(image_files)} Prophet plot(s)")
        else:
            self.prophet_status.configure(text="No Prophet images found")

    def refresh_mass_images(self):
        if not hasattr(self, 'mass_image_viewer'):
            return
            
        mass_patterns = [
            'methane_mass_map*.png',
            '*mass*.png'
        ]
        
        image_files = []
        for pattern in mass_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.mass_image_viewer.load_images(image_files)
            self.mass_status.configure(text=f"✅ Found {len(image_files)} mass estimation plot(s)")
        else:
            self.mass_status.configure(text="No mass estimation images found")

    def run_scatter_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.scatter_status.configure(text="🔄 Running scatter plot analysis...")
        self.run_scatter_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_scatter_analysis_thread)

    def _run_scatter_analysis_thread(self):
        try:
            generated_files, error = ScatterAnalysis.run_scatter_analysis(self.data_file)
            self.after(0, self._scatter_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._scatter_analysis_complete, [], str(e))

    def _scatter_analysis_complete(self, generated_files, error):
        self.run_scatter_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.scatter_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Scatter plot analysis failed:\n{error}")
        elif generated_files:
            self.scatter_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.scatter_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} scatter plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.scatter_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No scatter plots were generated. Please check your data file.")

    def run_histogram_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.histogram_status.configure(text="🔄 Running histogram analysis...")
        self.run_histogram_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_histogram_analysis_thread)

    def _run_histogram_analysis_thread(self):
        try:
            generated_files, error = HistogramAnalysis.run_histogram_analysis(self.data_file)
            self.after(0, self._histogram_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._histogram_analysis_complete, [], str(e))

    def _histogram_analysis_complete(self, generated_files, error):
        self.run_histogram_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.histogram_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Histogram analysis failed:\n{error}")
        elif generated_files:
            self.histogram_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.histogram_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} histograms:\n" + 
                              "\n".join(plot_names))
        else:
            self.histogram_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No histograms were generated. Please check your data file.")

    def run_box_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.box_status.configure(text="🔄 Running box plot analysis...")
        self.run_box_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_box_analysis_thread)

    def _run_box_analysis_thread(self):
        try:
            generated_files, error = BoxAnalysis.run_box_analysis(self.data_file)
            self.after(0, self._box_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._box_analysis_complete, [], str(e))

    def _box_analysis_complete(self, generated_files, error):
        self.run_box_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.box_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Box plot analysis failed:\n{error}")
        elif generated_files:
            self.box_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.box_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} box plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.box_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No box plots were generated. Please check your data file.")

    def run_contour_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.contour_status.configure(text="🔄 Running contour plot analysis...")
        self.run_contour_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_contour_analysis_thread)

    def _run_contour_analysis_thread(self):
        try:
            generated_files, error = ContourAnalysis.run_contour_analysis(self.data_file)
            self.after(0, self._contour_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._contour_analysis_complete, [], str(e))

    def _contour_analysis_complete(self, generated_files, error):
        self.run_contour_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.contour_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Contour plot analysis failed:\n{error}")
        elif generated_files:
            self.contour_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.contour_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} contour plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.contour_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No contour plots were generated. Please check your data file.")

    def run_kmeans_analysis(self, mode):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        status = getattr(self, f'kmeans_{mode}_status')
        btn = getattr(self, f'run_kmeans_{mode}_btn')
        
        status.configure(text=f"🔄 Running K-means analysis ({mode})...")
        btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_kmeans_analysis_thread, mode)

    def _run_kmeans_analysis_thread(self, mode):
        try:
            generated_files, error = KMeansAnalysis.run_kmeans_analysis(self.data_file, mode=mode)
            self.after(0, self._kmeans_analysis_complete, generated_files, error, mode)
        except Exception as e:
            self.after(0, self._kmeans_analysis_complete, [], str(e), mode)

    def _kmeans_analysis_complete(self, generated_files, error, mode):
        btn = getattr(self, f'run_kmeans_{mode}_btn')
        status = getattr(self, f'kmeans_{mode}_status')
        viewer = getattr(self, f'kmeans_{mode}_image_viewer')
        
        btn.configure(state="normal", text="Run Analysis")
        
        if error:
            status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"K-means analysis ({mode}) failed:\n{error}")
        elif generated_files:
            status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} K-means plots ({mode}):\n" + 
                              "\n".join(plot_names))
        else:
            status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", f"No K-means plots were generated ({mode}). Please check your data file.")

    def run_dfa_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        self.dfa_status.configure(text="🔄 Running DFA analysis...")
        self.run_dfa_btn.configure(state="disabled", text="Running...")

        self._start_thread(self._run_dfa_analysis_thread)

    def _run_dfa_analysis_thread(self):
        try:
            generated_files, error = DFAAnalysis.run_dfa_analysis(self.data_file)
            self.after(0, self._dfa_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._dfa_analysis_complete, [], str(e))

    def _dfa_analysis_complete(self, generated_files, error):
        self.run_dfa_btn.configure(state="normal", text="Run Analysis")
        if error:
            self.dfa_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"DFA analysis failed:\n{error}")
        elif generated_files:
            self.dfa_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.dfa_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", f"Generated DFA plot:\n{generated_files[0]}")
        else:
            self.dfa_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No DFA plot generated.")

    def run_psa_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        self.psa_status.configure(text="🔄 Running PSA analysis...")
        self.run_psa_btn.configure(state="disabled", text="Running...")
        self._start_thread(self._run_psa_analysis_thread)

    def _run_psa_analysis_thread(self):
        try:
            generated_files, error = PSAAnalysis.run_psa_analysis(self.data_file)
            self.after(0, self._psa_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._psa_analysis_complete, [], str(e))

    def _psa_analysis_complete(self, generated_files, error):
        self.run_psa_btn.configure(state="normal", text="Run Analysis")
        if error:
            self.psa_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"PSA analysis failed:\n{error}")
        elif generated_files:
            self.psa_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.psa_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", f"Generated PSA plots:\n" + "\n".join(generated_files))
        else:
            self.psa_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No PSA plot generated.")

    def run_lstm_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.lstm_status.configure(text="🔄 Running LSTM analysis...")
        self.run_lstm_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_lstm_analysis_thread)

    def _run_lstm_analysis_thread(self):
        try:
            generated_files, error = LSTMAnalysis.run_lstm_analysis(self.data_file)
            self.after(0, self._lstm_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._lstm_analysis_complete, [], str(e))

    def _lstm_analysis_complete(self, generated_files, error):
        self.run_lstm_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.lstm_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"LSTM analysis failed:\n{error}")
        elif generated_files:
            self.lstm_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.lstm_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} LSTM plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.lstm_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No LSTM plots were generated. Please check your data file.")

    def run_prophet_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.prophet_status.configure(text="🔄 Running Prophet analysis...")
        self.run_prophet_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_prophet_analysis_thread)

    def _run_prophet_analysis_thread(self):
        try:
            generated_files, error = ProphetAnalysis.run_prophet_analysis(self.data_file)
            self.after(0, self._prophet_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._prophet_analysis_complete, [], str(e))

    def _prophet_analysis_complete(self, generated_files, error):
        self.run_prophet_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.prophet_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Prophet analysis failed:\n{error}")
        elif generated_files:
            self.prophet_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.prophet_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} Prophet plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.prophet_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No Prophet plots were generated. Please check your data file.")

    def run_mass_estimation(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        try:
            df = pd.read_csv(self.data_file)
            required_columns = ['methane3', 'longitude', 'latitude']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                messagebox.showerror("Error", f"Data file missing required columns: {', '.join(missing)}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read data file: {str(e)}")
            return
        
        self.mass_status.configure(text="🔄 Running mass estimation...")
        self.run_mass_btn.configure(state="disabled", text="Running...")
        
        self._start_thread(self._run_mass_estimation_thread)

    def _run_mass_estimation_thread(self):
        try:
            generated_files, total_mass, error = MassEstimation.run_mass_estimation(self.data_file)
            self.after(0, self._mass_estimation_complete, generated_files, total_mass, error)
        except Exception as e:
            self.after(0, self._mass_estimation_complete, [], None, str(e))

    def _mass_estimation_complete(self, generated_files, total_mass, error):
        self.run_mass_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.mass_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Mass estimation failed:\n{error}")
        elif generated_files and total_mass is not None:
            self.mass_status.configure(text=f"✅ Total methane mass: {total_mass:.2f} tons")
            self.mass_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully estimated total methane mass: {total_mass:.2f} tons\n"
                              f"Generated plot: {generated_files[0]}")
        else:
            self.mass_status.configure(text="❌ No results generated")
            messagebox.showwarning("No Results", "No results were generated. Please check your data file.")

if __name__ == "__main__":
    app = MethaneAnalysisApp()
    app.mainloop()
