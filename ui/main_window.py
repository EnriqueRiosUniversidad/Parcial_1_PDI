"""Minimal Tkinter user interface for browsing medical images."""

from __future__ import annotations

from pathlib import Path
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.algorithms import apply_clahe, apply_histogram_equalization, apply_morphological_algorithm
from core.batch import (
    BatchConfig,
    append_image_ranking_csv,
    build_folder_comparison,
    build_image_comparison,
)
from core.image_loader import list_image_files, load_image
from core.histograms import calculate_grayscale_histogram
from core.metrics import (
    calculate_ambe,
    calculate_basic_metrics,
    calculate_psnr,
    evaluate_brightness_preservation,
    evaluate_contrast_change,
)


class ImageApp:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Medical Contrast Explorer")
        self.root.minsize(1100, 700)
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")

        self.current_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.selected_image: np.ndarray | None = None
        self.selected_image_name: str | None = None
        self.original_canvas: FigureCanvasTkAgg | None = None
        self.processed_canvas: FigureCanvasTkAgg | None = None
        self.histogram_canvas: FigureCanvasTkAgg | None = None
        self.processed_histogram_canvas: FigureCanvasTkAgg | None = None
        self.right_canvas: tk.Canvas | None = None
        self.right_scrollbar: ttk.Scrollbar | None = None
        self.active_scroll_canvas: tk.Canvas | None = None
        self.clip_limit_var = tk.StringVar(value="2.0")
        self.tile_grid_x_var = tk.StringVar(value="8")
        self.tile_grid_y_var = tk.StringVar(value="8")
        self.kernel_size_var = tk.StringVar(value="15")
        self.process_button: ttk.Button | None = None
        self.image_ranking_button: ttk.Button | None = None
        self.experiment_button: ttk.Button | None = None
        self.kernel_results_tree: ttk.Treeview | None = None
        self.global_comparison_tree: ttk.Treeview | None = None
        self.image_comparison_tree: ttk.Treeview | None = None
        self.comparison_refresh_button: ttk.Button | None = None
        self.global_comparison_frame: ttk.LabelFrame | None = None
        self.image_comparison_frame: ttk.LabelFrame | None = None
        self.experiment_frame: ttk.LabelFrame | None = None
        self.metrics_frame: ttk.LabelFrame | None = None
        self.single_image_ranking_csv: Path | None = None

        self._setup_visual_style()
        self._build_ui()

    def _setup_visual_style(self) -> None:
        """Apply a restrained fantasy-inspired dark theme."""
        self.bg_main = "#111915"
        self.bg_panel = "#17231d"
        self.bg_panel_alt = "#1d2a23"
        self.bg_canvas = "#0f1713"
        self.bg_border = "#33463a"
        self.text_main = "#e7e2d4"
        self.text_muted = "#b8c0b0"
        self.accent_gold = "#c9b46b"
        self.accent_gold_soft = "#8e7a3f"
        self.entry_bg = "#203028"

        self.root.configure(bg=self.bg_main)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.bg_main, foreground=self.text_main, fieldbackground=self.entry_bg)
        style.configure("TFrame", background=self.bg_main)
        style.configure("TLabelframe", background=self.bg_panel, foreground=self.accent_gold, borderwidth=1)
        style.configure("TLabelframe.Label", background=self.bg_panel, foreground=self.accent_gold, font=("Helvetica", 10, "bold"))
        style.configure("TLabel", background=self.bg_main, foreground=self.text_main, font=("Helvetica", 10))
        style.configure("Header.TLabel", background=self.bg_main, foreground=self.accent_gold, font=("Helvetica", 11, "bold"))
        style.configure("TButton", background=self.bg_panel_alt, foreground=self.text_main, padding=(10, 6), borderwidth=1, focusthickness=1, focuscolor=self.accent_gold_soft)
        style.map(
            "TButton",
            background=[("active", self.bg_border), ("pressed", self.bg_border)],
            foreground=[("disabled", self.text_muted)],
        )
        style.configure("TCombobox", fieldbackground=self.entry_bg, background=self.bg_panel_alt, foreground=self.text_main, arrowcolor=self.accent_gold)
        style.map("TCombobox", fieldbackground=[("readonly", self.entry_bg)], foreground=[("readonly", self.text_main)])
        style.configure("Treeview", background=self.bg_panel, fieldbackground=self.bg_panel, foreground=self.text_main, rowheight=24, bordercolor=self.bg_border, borderwidth=1)
        style.configure("Treeview.Heading", background=self.bg_panel_alt, foreground=self.text_main, relief="flat", font=("Helvetica", 10, "bold"))
        style.map("Treeview", background=[("selected", self.bg_border)], foreground=[("selected", self.text_main)])
        style.configure("Vertical.TScrollbar", background=self.bg_panel_alt, troughcolor=self.bg_main, arrowcolor=self.accent_gold)

    def _build_ui(self) -> None:
        """Create the base layout."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(10, 8, 10, 6), style="TFrame")
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        toolbar.columnconfigure(0, weight=1)
        toolbar.columnconfigure(1, weight=1)
        toolbar.columnconfigure(2, weight=1)

        source_frame = ttk.LabelFrame(toolbar, text="Origen", padding=(8, 6))
        source_frame.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        source_frame.columnconfigure(0, weight=1)
        select_button = ttk.Button(source_frame, text="Abrir carpeta", command=self.select_folder)
        select_button.grid(row=0, column=0, sticky="w")
        self.folder_label = ttk.Label(source_frame, text="Ninguna carpeta seleccionada")
        self.folder_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        algorithm_frame = ttk.LabelFrame(toolbar, text="Procesamiento", padding=(8, 6))
        algorithm_frame.grid(row=0, column=1, sticky="ew", padx=8)
        algorithm_frame.columnconfigure(1, weight=1)
        ttk.Label(algorithm_frame, text="Algoritmo:").grid(row=0, column=0, sticky="w")
        self.algorithm_var = tk.StringVar(value="HE")
        self.algorithm_options = [
            "HE",
            "CLAHE",
            "White Top-Hat",
            "Black Top-Hat",
            "Enhanced Top-Hat",
        ]
        self.algorithm_combo = ttk.Combobox(
            algorithm_frame,
            textvariable=self.algorithm_var,
            values=self.algorithm_options,
            state="readonly",
            width=20,
        )
        self.algorithm_combo.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)
        self.process_button = ttk.Button(algorithm_frame, text="Procesar", command=self.process_current_image)
        self.process_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.image_ranking_button = ttk.Button(
            algorithm_frame,
            text="Calcular ranking imagen",
            command=self.calculate_selected_image_ranking,
        )
        self.image_ranking_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.comparison_refresh_button = ttk.Button(
            algorithm_frame,
            text="Obtener datos globales",
            command=self.show_global_comparison_view,
        )
        self.comparison_refresh_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        params_frame = ttk.LabelFrame(toolbar, text="Parámetros", padding=(8, 6))
        params_frame.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        params_frame.columnconfigure(1, weight=1)
        ttk.Label(params_frame, text="CLAHE clipLimit").grid(row=0, column=0, sticky="w")
        self.clip_limit_entry = ttk.Entry(params_frame, textvariable=self.clip_limit_var, width=8)
        self.clip_limit_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(params_frame, text="CLAHE tileGrid").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tile_frame = ttk.Frame(params_frame)
        tile_frame.grid(row=1, column=1, sticky="w", pady=(6, 0))
        self.tile_grid_x_entry = ttk.Entry(tile_frame, textvariable=self.tile_grid_x_var, width=4)
        self.tile_grid_x_entry.grid(row=0, column=0)
        ttk.Label(tile_frame, text="x").grid(row=0, column=1, padx=4)
        self.tile_grid_y_entry = ttk.Entry(tile_frame, textvariable=self.tile_grid_y_var, width=4)
        self.tile_grid_y_entry.grid(row=0, column=2)
        ttk.Label(params_frame, text="Top-Hat kernel").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.kernel_size_entry = ttk.Entry(params_frame, textvariable=self.kernel_size_var, width=8)
        self.kernel_size_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0))

        left_panel = ttk.Frame(self.root, padding=(10, 0, 5, 8))
        left_panel.grid(row=1, column=0, sticky="nsew")
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(1, weight=1)

        ttk.Label(left_panel, text="Imágenes detectadas").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )

        list_frame = ttk.Frame(left_panel)
        list_frame.grid(row=1, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.image_listbox = tk.Listbox(
            list_frame,
            activestyle="dotbox",
            bg=self.bg_panel,
            fg=self.text_main,
            selectbackground=self.accent_gold_soft,
            selectforeground=self.text_main,
            highlightthickness=1,
            highlightbackground=self.bg_border,
            highlightcolor=self.accent_gold,
            relief="flat",
            borderwidth=0,
        )
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.image_listbox.configure(yscrollcommand=scrollbar.set)

        right_panel = ttk.Frame(self.root, padding=(5, 0, 10, 8))
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_panel, highlightthickness=0, bg=self.bg_main, bd=0)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")

        self.right_scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.right_canvas.yview)
        self.right_scrollbar.grid(row=0, column=1, sticky="ns")
        self.right_canvas.configure(yscrollcommand=self.right_scrollbar.set)

        preview_frame = ttk.Frame(self.right_canvas)
        preview_window = self.right_canvas.create_window((0, 0), window=preview_frame, anchor="nw")

        def _update_scrollregion(event: tk.Event) -> None:
            self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

        def _sync_width(event: tk.Event) -> None:
            self.right_canvas.itemconfigure(preview_window, width=event.width)

        preview_frame.bind("<Configure>", _update_scrollregion)
        self.right_canvas.bind("<Configure>", _sync_width)
        self._bind_mousewheel(self.root, self.right_canvas)
        self._register_scroll_target(preview_frame)

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.rowconfigure(3, weight=1)
        preview_frame.rowconfigure(4, weight=0)

        ttk.Label(preview_frame, text="Histograma original").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(preview_frame, text="Imagen original").grid(row=0, column=1, sticky="w", pady=(0, 5))

        self.histogram_container = tk.Frame(preview_frame, bd=1, relief="solid", bg=self.bg_panel)
        self.histogram_container.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 6))
        self.histogram_container.columnconfigure(0, weight=1)
        self.histogram_container.rowconfigure(0, weight=1)

        self.original_container = tk.Frame(preview_frame, bd=1, relief="solid", bg=self.bg_panel)
        self.original_container.grid(row=1, column=1, sticky="nsew", pady=(0, 6))
        self.original_container.columnconfigure(0, weight=1)
        self.original_container.rowconfigure(0, weight=1)

        ttk.Label(preview_frame, text="Histograma procesado").grid(row=2, column=0, sticky="w", pady=(0, 5))
        ttk.Label(preview_frame, text="Imagen procesada").grid(row=2, column=1, sticky="w", pady=(0, 5))

        self.processed_histogram_container = tk.Frame(preview_frame, bd=1, relief="solid", bg=self.bg_panel)
        self.processed_histogram_container.grid(row=3, column=0, sticky="nsew", padx=(0, 8), pady=(0, 6))
        self.processed_histogram_container.columnconfigure(0, weight=1)
        self.processed_histogram_container.rowconfigure(0, weight=1)

        self.processed_container = tk.Frame(preview_frame, bd=1, relief="solid", bg=self.bg_panel)
        self.processed_container.grid(row=3, column=1, sticky="nsew", pady=(0, 6))
        self.processed_container.columnconfigure(0, weight=1)
        self.processed_container.rowconfigure(0, weight=1)

        self.metrics_frame = ttk.LabelFrame(preview_frame, text="Métricas comparativas", padding=(8, 4))
        self.metrics_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.metrics_frame.columnconfigure(1, weight=1)
        self.metrics_frame.grid_remove()

        ttk.Label(self.metrics_frame, text="Desviación estándar original:").grid(row=0, column=0, sticky="w")
        self.original_std_value_label = ttk.Label(self.metrics_frame, text="-")
        self.original_std_value_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(self.metrics_frame, text="Desviación estándar procesada:").grid(row=1, column=0, sticky="w")
        self.processed_std_value_label = ttk.Label(self.metrics_frame, text="-")
        self.processed_std_value_label.grid(row=1, column=1, sticky="w", padx=(10, 0))

        ttk.Label(self.metrics_frame, text="AMBE:").grid(row=2, column=0, sticky="w")
        self.ambe_value_label = ttk.Label(self.metrics_frame, text="-")
        self.ambe_value_label.grid(row=2, column=1, sticky="w", padx=(10, 0))

        ttk.Label(self.metrics_frame, text="PSNR:").grid(row=3, column=0, sticky="w")
        self.psnr_value_label = ttk.Label(self.metrics_frame, text="-")
        self.psnr_value_label.grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.evaluation_label = ttk.Label(preview_frame, text="Evaluación: -", wraplength=620, justify="left")
        self.evaluation_label.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self.helper_label = ttk.Label(
            preview_frame,
            text="HE: sin parámetros. CLAHE: usa clipLimit y tileGridSize. Morfología: usa kernel.",
            wraplength=620,
            justify="left",
        )
        self.helper_label.grid(row=6, column=0, columnspan=2, sticky="w", pady=(0, 4))

        self.experiment_frame = ttk.LabelFrame(preview_frame, text="Experimento Top-Hat", padding=(8, 4))
        self.experiment_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.experiment_frame.grid_remove()
        self.experiment_frame.columnconfigure(0, weight=1)
        self.experiment_button = ttk.Button(
            self.experiment_frame,
            text="Probar kernels 3-5-7-9",
            command=self.run_top_hat_kernel_experiment,
        )
        self.experiment_button.grid(row=0, column=0, sticky="ew")

        tree_frame = ttk.Frame(self.experiment_frame)
        tree_frame.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        columns = ("kernel", "std", "ambe", "psnr")
        self.kernel_results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=5)
        self.kernel_results_tree.heading("kernel", text="Kernel")
        self.kernel_results_tree.heading("std", text="Std. dev.")
        self.kernel_results_tree.heading("ambe", text="AMBE")
        self.kernel_results_tree.heading("psnr", text="PSNR")
        self.kernel_results_tree.column("kernel", width=70, anchor="center")
        self.kernel_results_tree.column("std", width=90, anchor="center")
        self.kernel_results_tree.column("ambe", width=90, anchor="center")
        self.kernel_results_tree.column("psnr", width=90, anchor="center")
        self.kernel_results_tree.grid(row=0, column=0, sticky="nsew")

        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.kernel_results_tree.yview)
        tree_scrollbar.grid(row=0, column=1, sticky="ns")
        self.kernel_results_tree.configure(yscrollcommand=tree_scrollbar.set)

        self.global_comparison_frame = ttk.LabelFrame(preview_frame, text="Comparativa global", padding=(8, 4))
        self.global_comparison_frame.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.global_comparison_frame.grid_remove()
        self.global_comparison_frame.columnconfigure(0, weight=1)

        global_table_frame = ttk.Frame(self.global_comparison_frame)
        global_table_frame.grid(row=0, column=0, sticky="nsew")
        global_table_frame.columnconfigure(0, weight=1)
        global_table_frame.rowconfigure(0, weight=1)

        global_columns = ("rank", "algorithm", "avg_processed_std", "avg_ambe", "avg_psnr", "ranking_score")
        self.global_comparison_tree = ttk.Treeview(global_table_frame, columns=global_columns, show="headings", height=6)
        for column, title, width in [
            ("rank", "#", 40),
            ("algorithm", "Algoritmo", 180),
            ("avg_processed_std", "Std proc.", 80),
            ("avg_ambe", "AMBE", 80),
            ("avg_psnr", "PSNR", 80),
            ("ranking_score", "Score", 80),
        ]:
            self.global_comparison_tree.heading(column, text=title)
            self.global_comparison_tree.column(column, width=width, anchor="center")
        self.global_comparison_tree.grid(row=0, column=0, sticky="nsew")
        global_scroll = ttk.Scrollbar(global_table_frame, orient="vertical", command=self.global_comparison_tree.yview)
        global_scroll.grid(row=0, column=1, sticky="ns")
        self.global_comparison_tree.configure(yscrollcommand=global_scroll.set)

        self.image_comparison_frame = ttk.LabelFrame(preview_frame, text="Ranking de imagen", padding=(8, 4))
        self.image_comparison_frame.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.image_comparison_frame.grid_remove()
        self.image_comparison_frame.columnconfigure(0, weight=1)

        image_table_frame = ttk.Frame(self.image_comparison_frame)
        image_table_frame.grid(row=0, column=0, sticky="nsew")
        image_table_frame.columnconfigure(0, weight=1)
        image_table_frame.rowconfigure(0, weight=1)

        image_columns = ("algorithm", "original_std", "processed_std", "ambe", "psnr")
        self.image_comparison_tree = ttk.Treeview(image_table_frame, columns=image_columns, show="headings", height=6)
        for column, title, width in [
            ("algorithm", "Algoritmo", 180),
            ("original_std", "Std orig.", 80),
            ("processed_std", "Std proc.", 80),
            ("ambe", "AMBE", 80),
            ("psnr", "PSNR", 80),
        ]:
            self.image_comparison_tree.heading(column, text=title)
            self.image_comparison_tree.column(column, width=width, anchor="center")
        self.image_comparison_tree.grid(row=0, column=0, sticky="nsew")
        image_scroll = ttk.Scrollbar(image_table_frame, orient="vertical", command=self.image_comparison_tree.yview)
        image_scroll.grid(row=0, column=1, sticky="ns")
        self.image_comparison_tree.configure(yscrollcommand=image_scroll.set)

        if self.global_comparison_tree is not None:
            self.global_comparison_tree.tag_configure("odd", background=self.bg_panel_alt)
            self.global_comparison_tree.tag_configure("even", background=self.bg_panel)
        if self.image_comparison_tree is not None:
            self.image_comparison_tree.tag_configure("odd", background=self.bg_panel_alt)
            self.image_comparison_tree.tag_configure("even", background=self.bg_panel)

        self.status_label = ttk.Label(self.root, text="Listo", anchor="w", style="Header.TLabel")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self._show_placeholder(self.original_container, "Selecciona una carpeta para comenzar")
        self._show_placeholder(self.processed_container, "La imagen procesada aparecerá aquí")
        self._show_placeholder(self.histogram_container, "Selecciona una imagen para ver su histograma")
        self._show_placeholder(self.processed_histogram_container, "El histograma procesado aparecerá aquí")
        self._update_parameter_visibility()
        self._update_metrics_visibility(False)
        self._update_experiment_visibility()

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def select_folder(self) -> None:
        """Open a folder picker and load supported images."""
        project_dir = Path(__file__).resolve().parent.parent
        selected_folder = filedialog.askdirectory(
            title="Selecciona la carpeta con imágenes JPG",
            initialdir=project_dir,
        )
        if not selected_folder:
            return

        folder_path = Path(selected_folder)
        self.current_folder = folder_path
        self.folder_label.configure(text=str(folder_path))
        self.single_image_ranking_csv = Path(__file__).resolve().parent.parent / "results" / f"{folder_path.name}_ranking_imagen.csv"

        self.image_paths = list_image_files(folder_path)
        self._refresh_image_list()

        if self.image_paths:
            self.status_label.configure(text=f"{len(self.image_paths)} imagen(es) detectada(s)")
            self._clear_selection_and_views()
        else:
            self.status_label.configure(text="No se encontraron imágenes compatibles")
            self._show_placeholder("La carpeta no contiene imágenes compatibles")

    def _refresh_image_list(self) -> None:
        """Update the listbox with discovered image names."""
        self.image_listbox.delete(0, tk.END)
        for image_path in self.image_paths:
            self.image_listbox.insert(tk.END, image_path.name)

    def on_image_selected(self, event: tk.Event | None = None) -> None:
        """Display the selected image in the preview area."""
        selection = self.image_listbox.curselection()
        if not selection:
            return

        selected_index = selection[0]
        image_path = self.image_paths[selected_index]
        image = load_image(image_path)

        if image is None:
            messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{image_path.name}")
            self.status_label.configure(text="Error al cargar la imagen")
            self._show_placeholder(self.original_container, "No se pudo cargar la imagen")
            self._show_placeholder(self.processed_container, "No se pudo cargar la imagen")
            self._show_placeholder(self.histogram_container, "No se pudo cargar la imagen")
            self._show_placeholder(self.processed_histogram_container, "No se pudo cargar la imagen")
            self._set_comparative_metrics(None)
            return

        self.selected_image = image
        self.selected_image_name = image_path.name
        self._show_image(image, image_path.name)
        self._show_placeholder(self.processed_container, "Pulsa Procesar para ver el resultado")
        self._show_placeholder(self.processed_histogram_container, "Pulsa Procesar para ver el histograma")
        self._show_histogram(image, image_path.name, self.histogram_container)
        self._set_comparative_metrics(None)
        self.status_label.configure(text=f"Imagen cargada: {image_path.name}")
        if self.experiment_frame is not None:
            self.experiment_frame.grid_remove()
        if self.image_comparison_frame is not None:
            self.image_comparison_frame.grid_remove()
        if self.image_comparison_tree is not None:
            for item in self.image_comparison_tree.get_children():
                self.image_comparison_tree.delete(item)

    def _clear_preview(self, container: tk.Widget) -> None:
        """Remove the current preview widget if it exists."""
        if container is self.original_container and self.original_canvas is not None:
            self.original_canvas.get_tk_widget().destroy()
            self.original_canvas = None
        if container is self.processed_container and self.processed_canvas is not None:
            self.processed_canvas.get_tk_widget().destroy()
            self.processed_canvas = None
        if container is self.histogram_container and self.histogram_canvas is not None:
            self.histogram_canvas.get_tk_widget().destroy()
            self.histogram_canvas = None
        if container is self.processed_histogram_container and self.processed_histogram_canvas is not None:
            self.processed_histogram_canvas.get_tk_widget().destroy()
            self.processed_histogram_canvas = None

    def _clear_selection_and_views(self) -> None:
        """Clear selection-dependent views when a folder is loaded."""
        self.selected_image = None
        self.selected_image_name = None
        self._show_placeholder(self.original_container, "Selecciona una imagen para comenzar")
        self._show_placeholder(self.processed_container, "Selecciona una imagen para ver el resultado")
        self._show_placeholder(self.histogram_container, "Selecciona una imagen para ver su histograma")
        self._show_placeholder(self.processed_histogram_container, "Selecciona una imagen para ver el histograma procesado")
        self._set_comparative_metrics(None)
        self._update_metrics_visibility(False)
        if self.image_comparison_tree is not None:
            for item in self.image_comparison_tree.get_children():
                self.image_comparison_tree.delete(item)
        if self.experiment_frame is not None:
            self.experiment_frame.grid_remove()
        if self.global_comparison_frame is not None:
            self.global_comparison_frame.grid_remove()
        if self.image_comparison_frame is not None:
            self.image_comparison_frame.grid_remove()

    def _show_placeholder(self, container: tk.Widget, text: str) -> None:
        """Show a simple text placeholder in the preview area."""
        self._clear_preview(container)
        for child in container.winfo_children():
            child.destroy()

        label = ttk.Label(container, text=text, anchor="center", justify="center", style="Header.TLabel")
        label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def _show_image(self, image_bgr, title: str) -> None:
        """Render an image on the embedded Matplotlib canvas."""
        self._clear_preview(self.original_container)
        for child in self.original_container.winfo_children():
            child.destroy()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        figure = Figure(figsize=(4.6, 4.6), dpi=100)
        axes = figure.add_subplot(111)
        axes.imshow(image_rgb)
        axes.axis("off")
        figure.tight_layout(pad=0.2)

        self.original_canvas = FigureCanvasTkAgg(figure, master=self.original_container)
        widget = self.original_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        widget.bind("<Enter>", lambda event: setattr(self, "active_scroll_canvas", self.right_canvas), add="+")
        self.original_canvas.draw()

    def _show_processed_image(self, image_gray: np.ndarray, title: str) -> None:
        """Render the processed grayscale image."""
        self._clear_preview(self.processed_container)
        for child in self.processed_container.winfo_children():
            child.destroy()

        figure = Figure(figsize=(4.6, 4.6), dpi=100)
        axes = figure.add_subplot(111)
        axes.imshow(image_gray, cmap="gray")
        axes.axis("off")
        figure.tight_layout(pad=0.2)

        self.processed_canvas = FigureCanvasTkAgg(figure, master=self.processed_container)
        widget = self.processed_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        widget.bind("<Enter>", lambda event: setattr(self, "active_scroll_canvas", self.right_canvas), add="+")
        self.processed_canvas.draw()

    def _show_histogram(self, image_bgr: np.ndarray, title: str, container: tk.Widget) -> None:
        """Render the grayscale histogram for the selected image."""
        self._clear_preview(container)
        for child in container.winfo_children():
            child.destroy()

        histogram = calculate_grayscale_histogram(image_bgr)

        figure = Figure(figsize=(4.6, 2.2), dpi=100)
        axes = figure.add_subplot(111)
        axes.plot(histogram, color="black", linewidth=1)
        axes.set_xlim([0, 255])
        axes.set_title("Histograma", fontsize=10)
        axes.set_xlabel("Intensidad")
        axes.set_ylabel("Frecuencia")
        axes.grid(alpha=0.2)
        figure.tight_layout(pad=0.6)

        canvas = FigureCanvasTkAgg(figure, master=container)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        widget.bind("<Enter>", lambda event: setattr(self, "active_scroll_canvas", self.right_canvas), add="+")
        canvas.draw()

        if container is self.histogram_container:
            self.histogram_canvas = canvas
        elif container is self.processed_histogram_container:
            self.processed_histogram_canvas = canvas

    def _bind_mousewheel(self, root: tk.Widget, canvas: tk.Canvas) -> None:
        """Enable mouse wheel scrolling anywhere over the app."""
        self.active_scroll_canvas = canvas

        def _on_mousewheel(event: tk.Event) -> str:
            target_canvas = self.active_scroll_canvas or canvas
            if event.delta:
                target_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _on_linux_scroll_up(event: tk.Event) -> str:
            target_canvas = self.active_scroll_canvas or canvas
            target_canvas.yview_scroll(-1, "units")
            return "break"

        def _on_linux_scroll_down(event: tk.Event) -> str:
            target_canvas = self.active_scroll_canvas or canvas
            target_canvas.yview_scroll(1, "units")
            return "break"

        root.bind_all("<MouseWheel>", _on_mousewheel)
        root.bind_all("<Button-4>", _on_linux_scroll_up)
        root.bind_all("<Button-5>", _on_linux_scroll_down)

    def _register_scroll_target(self, widget: tk.Widget) -> None:
        """Mark a widget subtree as part of the scrollable panel."""
        def _set_target(event: tk.Event) -> None:
            self.active_scroll_canvas = self.right_canvas

        def _clear_target(event: tk.Event) -> None:
            self.active_scroll_canvas = self.right_canvas

        widget.bind("<Enter>", _set_target, add="+")
        widget.bind("<Leave>", _clear_target, add="+")
        for child in widget.winfo_children():
            self._register_scroll_target(child)

    def _process_selected_algorithm(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply the selected enhancement algorithm."""
        algorithm_name = self.algorithm_var.get()
        if algorithm_name == "HE":
            return apply_histogram_equalization(image_bgr)
        if algorithm_name == "CLAHE":
            return apply_clahe(
                image_bgr,
                clip_limit=self._read_clip_limit(),
                tile_grid_size=self._read_tile_grid_size(),
            )
        if algorithm_name in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}:
            return apply_morphological_algorithm(
                image_bgr,
                algorithm_name,
                kernel_size=self._read_kernel_size(),
            )
        return apply_histogram_equalization(image_bgr)

    def _read_clip_limit(self) -> float:
        """Read the clip limit parameter with a safe default."""
        try:
            value = float(self.clip_limit_var.get())
            return value if value > 0 else 2.0
        except ValueError:
            return 2.0

    def _read_tile_grid_size(self) -> tuple[int, int]:
        """Read CLAHE tile grid size with safe defaults."""
        try:
            x_value = max(1, int(self.tile_grid_x_var.get()))
            y_value = max(1, int(self.tile_grid_y_var.get()))
            return (x_value, y_value)
        except ValueError:
            return (8, 8)

    def _read_kernel_size(self) -> int:
        """Read morphological kernel size with a safe default."""
        try:
            value = int(self.kernel_size_var.get())
            return value if value > 0 else 15
        except ValueError:
            return 15

    def _on_algorithm_change(self, event: tk.Event | None = None) -> None:
        """Mark the preview as needing reprocessing."""
        self._update_parameter_visibility()
        self._update_experiment_visibility()
        if self.selected_image is not None and self.selected_image_name is not None:
            self.status_label.configure(
                text=f"Algoritmo cambiado a {self.algorithm_var.get()}. Pulsa Procesar para actualizar."
            )

    def _update_parameter_visibility(self) -> None:
        """Enable only the parameter controls relevant to the selected algorithm."""
        algorithm_name = self.algorithm_var.get()
        is_clahe = algorithm_name == "CLAHE"
        is_morphology = algorithm_name in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}

        for widget in (self.clip_limit_entry, self.tile_grid_x_entry, self.tile_grid_y_entry):
            widget.configure(state="normal" if is_clahe else "disabled")

        self.kernel_size_entry.configure(state="normal" if is_morphology else "disabled")

        if is_clahe:
            self.helper_label.configure(text="CLAHE activo: ajusta clipLimit y tileGridSize, luego pulsa Procesar.")
        elif is_morphology:
            self.helper_label.configure(text="Top-Hat activo: ajusta kernel y pulsa Procesar.")
        else:
            self.helper_label.configure(text="HE activo: no requiere parámetros adicionales. Pulsa Procesar.")

    def _update_experiment_visibility(self, processed: bool = False) -> None:
        """Show the kernel experiment only after processing a Top-Hat image."""
        is_morphology = self.algorithm_var.get() in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}
        state = "normal" if is_morphology else "disabled"
        if self.experiment_button is not None:
            self.experiment_button.configure(state=state)
        if self.experiment_frame is not None:
            if is_morphology and processed and self.selected_image is not None:
                self.experiment_frame.grid()
            else:
                self.experiment_frame.grid_remove()

    def _update_metrics_visibility(self, visible: bool) -> None:
        """Show comparative metrics only after a processed image exists."""
        if self.metrics_frame is None:
            return
        if visible:
            self.metrics_frame.grid()
        else:
            self.metrics_frame.grid_remove()

    def process_current_image(self) -> None:
        """Process the currently selected image using the chosen algorithm."""
        if self.selected_image is None or self.selected_image_name is None:
            messagebox.showinfo("Procesar", "Primero selecciona una imagen.")
            return

        processed_image = self._process_selected_algorithm(self.selected_image)
        algorithm_name = self.algorithm_var.get()

        self._show_image(self.selected_image, self.selected_image_name)
        self._show_processed_image(processed_image, f"{algorithm_name}: {self.selected_image_name}")
        self._show_histogram(self.selected_image, self.selected_image_name, self.histogram_container)
        self._show_histogram(
            processed_image,
            f"{algorithm_name}: {self.selected_image_name}",
            self.processed_histogram_container,
        )
        self._set_comparative_metrics(self.selected_image, processed_image)
        self._update_metrics_visibility(True)
        self.status_label.configure(
            text=f"Procesado con {algorithm_name}: {self.selected_image_name}"
        )
        self._update_experiment_visibility(processed=True)

    def show_global_comparison_view(self) -> None:
        """Show and populate the global comparison view."""
        if self.current_folder is None:
            messagebox.showinfo("Comparativa", "Primero selecciona una carpeta.")
            return
        self.refresh_global_comparison_view()

    def run_batch_processing(self) -> None:
        """Process all JPG images in the selected folder with all algorithms."""
        if self.current_folder is None:
            messagebox.showinfo("Batch", "Primero selecciona una carpeta.")
            return

        try:
            result_paths = run_folder_batch(
                self.current_folder,
                Path(__file__).resolve().parent.parent / "results",
                BatchConfig(),
            )
        except Exception as exc:  # pragma: no cover - user-facing error path
            messagebox.showerror("Batch", f"No se pudo ejecutar el procesamiento batch:\n{exc}")
            return

        messagebox.showinfo(
            "Batch completado",
            "Procesamiento batch finalizado.\n"
            f"Carpeta: {result_paths['batch_dir']}\n"
            f"Comparativa global: {result_paths['comparison_csv']}",
        )
        self.status_label.configure(text=f"Batch completado para {self.current_folder.name}")

    def refresh_global_comparison_view(self) -> None:
        """Refresh the global comparison table for the current folder."""
        if self.current_folder is None or self.global_comparison_tree is None:
            return

        global_rows, _ = build_folder_comparison(self.current_folder, BatchConfig())
        if self.global_comparison_frame is not None:
            self.global_comparison_frame.grid()
        for item in self.global_comparison_tree.get_children():
            self.global_comparison_tree.delete(item)
        for row in global_rows:
            self.global_comparison_tree.insert(
                "",
                "end",
                values=(
                    row["rank"],
                    row["algorithm"],
                    row["avg_processed_std"],
                    row["avg_ambe"],
                    row["avg_psnr"],
                    row["ranking_score"],
                ),
                tags=("odd" if row["rank"] % 2 else "even",),
            )

    def refresh_image_comparison_view(self) -> None:
        """Refresh the per-image comparison table for the selected image."""
        if self.current_folder is None or self.selected_image_name is None or self.image_comparison_tree is None:
            return

        rows = (
            build_image_comparison(self.selected_image, self._current_batch_config())
            if self.selected_image is not None
            else []
        )
        if self.image_comparison_frame is not None:
            self.image_comparison_frame.grid()
        for item in self.image_comparison_tree.get_children():
            self.image_comparison_tree.delete(item)

        for row in rows:
            self.image_comparison_tree.insert(
                "",
                "end",
                values=(
                    row["algorithm"],
                    row["original_std"],
                    row["processed_std"],
                    row["ambe"],
                    row["psnr"],
                ),
                tags=("odd" if row["rank"] % 2 else "even",),
            )

    def calculate_selected_image_ranking(self) -> None:
        """Calculate and save the ranking for the currently selected image."""
        if self.selected_image is None or self.selected_image_name is None or self.single_image_ranking_csv is None:
            messagebox.showinfo("Ranking", "Primero selecciona una imagen.")
            return

        rows = build_image_comparison(self.selected_image, self._current_batch_config())
        append_image_ranking_csv(self.single_image_ranking_csv, self.selected_image_name, rows)
        if self.image_comparison_frame is not None:
            self.image_comparison_frame.grid()
        self.refresh_image_comparison_view()
        self.status_label.configure(
            text=f"Ranking calculado y guardado para {self.selected_image_name}"
        )

    def run_top_hat_kernel_experiment(self) -> None:
        """Run the selected Top-Hat algorithm across multiple kernel sizes."""
        if self.selected_image is None or self.selected_image_name is None:
            messagebox.showinfo("Experimento", "Primero selecciona una imagen.")
            return

        algorithm_name = self.algorithm_var.get()
        if algorithm_name not in {"White Top-Hat", "Black Top-Hat", "Enhanced Top-Hat"}:
            messagebox.showinfo("Experimento", "Selecciona un algoritmo Top-Hat para ejecutar esta prueba.")
            return

        kernel_sizes = [3, 5, 7, 9]
        original_image = self.selected_image
        original_metrics = calculate_basic_metrics(original_image)

        if self.kernel_results_tree is not None:
            for item in self.kernel_results_tree.get_children():
                self.kernel_results_tree.delete(item)

        results_rows: list[dict[str, object]] = []
        for kernel_size in kernel_sizes:
            processed_image = apply_morphological_algorithm(
                original_image,
                algorithm_name,
                kernel_size=kernel_size,
            )
            processed_metrics = calculate_basic_metrics(processed_image)
            ambe_value = calculate_ambe(original_image, processed_image)
            psnr_value = calculate_psnr(original_image, processed_image)
            results_rows.append(
                {
                    "image_name": self.selected_image_name,
                    "algorithm": algorithm_name,
                    "kernel_size": kernel_size,
                    "original_std": round(original_metrics["desviacion_estandar"], 4),
                    "processed_std": round(processed_metrics["desviacion_estandar"], 4),
                    "ambe": round(ambe_value, 4),
                    "psnr": "Inf" if np.isinf(psnr_value) else round(psnr_value, 4),
                }
            )

            if self.kernel_results_tree is not None:
                self.kernel_results_tree.insert(
                    "",
                    "end",
                    values=(
                        f"{kernel_size}x{kernel_size}",
                        f"{processed_metrics['desviacion_estandar']:.2f}",
                        f"{ambe_value:.2f}",
                        "Inf" if np.isinf(psnr_value) else f"{psnr_value:.2f}",
                    ),
                )

        self._save_kernel_experiment_csv(results_rows)
        self.status_label.configure(
            text=f"Experimento guardado en results/ para {self.selected_image_name}"
        )

    def _save_kernel_experiment_csv(self, rows: list[dict[str, object]]) -> None:
        """Persist kernel experiment results to CSV in the results folder."""
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        image_label = self.selected_image_name or "image"
        safe_name = image_label.replace("/", "_").replace("\\", "_").replace(" ", "_")
        csv_path = results_dir / f"{Path(safe_name).stem}_top_hat_kernels.csv"

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "image_name",
                    "algorithm",
                    "kernel_size",
                    "original_std",
                    "processed_std",
                    "ambe",
                    "psnr",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _set_comparative_metrics(self, original_image: np.ndarray | None, processed_image: np.ndarray | None = None) -> None:
        """Update comparative metric labels and evaluation text."""
        if original_image is None or processed_image is None:
            self.original_std_value_label.configure(text="-")
            self.processed_std_value_label.configure(text="-")
            self.ambe_value_label.configure(text="-")
            self.psnr_value_label.configure(text="-")
            self.evaluation_label.configure(text="Evaluación: -")
            return

        original_metrics = calculate_basic_metrics(original_image)
        processed_metrics = calculate_basic_metrics(processed_image)
        ambe_value = calculate_ambe(original_image, processed_image)
        psnr_value = calculate_psnr(original_image, processed_image)

        self.original_std_value_label.configure(text=f"{original_metrics['desviacion_estandar']:.2f}")
        self.processed_std_value_label.configure(text=f"{processed_metrics['desviacion_estandar']:.2f}")
        self.ambe_value_label.configure(text=f"{ambe_value:.2f}")
        self.psnr_value_label.configure(text="Inf" if np.isinf(psnr_value) else f"{psnr_value:.2f}")

        contrast_text = evaluate_contrast_change(
            original_metrics["desviacion_estandar"],
            processed_metrics["desviacion_estandar"],
        )
        brightness_text = evaluate_brightness_preservation(ambe_value)
        self.evaluation_label.configure(text=f"Evaluación: {contrast_text} {brightness_text}")

    def _current_batch_config(self) -> BatchConfig:
        """Build a batch-style config from the current UI parameter values."""
        return BatchConfig(
            clahe_clip_limit=self._read_clip_limit(),
            clahe_tile_grid_size=self._read_tile_grid_size(),
            top_hat_kernel_size=self._read_kernel_size(),
        )
