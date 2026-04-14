"""Minimal Tkinter user interface for browsing medical images."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.algorithms import apply_clahe, apply_histogram_equalization, apply_morphological_algorithm
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
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

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
        self.clip_limit_var = tk.StringVar(value="2.0")
        self.tile_grid_x_var = tk.StringVar(value="8")
        self.tile_grid_y_var = tk.StringVar(value="8")
        self.kernel_size_var = tk.StringVar(value="15")
        self.process_button: ttk.Button | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Create the base layout."""
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        toolbar.columnconfigure(1, weight=1)

        select_button = ttk.Button(
            toolbar,
            text="Seleccionar carpeta",
            command=self.select_folder,
        )
        select_button.grid(row=0, column=0, sticky="w")

        ttk.Label(toolbar, text="Algoritmo:").grid(row=0, column=1, sticky="e", padx=(20, 5))
        self.algorithm_var = tk.StringVar(value="HE")
        self.algorithm_options = [
            "HE",
            "CLAHE",
            "White Top-Hat",
            "Black Top-Hat",
            "Enhanced Top-Hat",
        ]
        self.algorithm_combo = ttk.Combobox(
            toolbar,
            textvariable=self.algorithm_var,
            values=self.algorithm_options,
            state="readonly",
            width=18,
        )
        self.algorithm_combo.grid(row=0, column=2, sticky="w")
        self.algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)

        self.process_button = ttk.Button(toolbar, text="Procesar", command=self.process_current_image)
        self.process_button.grid(row=0, column=3, sticky="w", padx=(20, 0))

        ttk.Label(toolbar, text="clipLimit:").grid(row=1, column=0, sticky="e", pady=(8, 0))
        self.clip_limit_entry = ttk.Entry(toolbar, textvariable=self.clip_limit_var, width=8)
        self.clip_limit_entry.grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(toolbar, text="tileGridSize:").grid(row=1, column=2, sticky="e", padx=(20, 5), pady=(8, 0))
        tile_frame = ttk.Frame(toolbar)
        tile_frame.grid(row=1, column=3, sticky="w", pady=(8, 0))
        self.tile_grid_x_entry = ttk.Entry(tile_frame, textvariable=self.tile_grid_x_var, width=4)
        self.tile_grid_x_entry.grid(row=0, column=0)
        ttk.Label(tile_frame, text="x").grid(row=0, column=1, padx=4)
        self.tile_grid_y_entry = ttk.Entry(tile_frame, textvariable=self.tile_grid_y_var, width=4)
        self.tile_grid_y_entry.grid(row=0, column=2)

        ttk.Label(toolbar, text="kernel:").grid(row=2, column=0, sticky="e", pady=(8, 0))
        self.kernel_size_entry = ttk.Entry(toolbar, textvariable=self.kernel_size_var, width=8)
        self.kernel_size_entry.grid(row=2, column=1, sticky="w", pady=(8, 0))

        self.folder_label = ttk.Label(toolbar, text="Ninguna carpeta seleccionada")
        self.folder_label.grid(row=0, column=4, sticky="w", padx=(10, 0))

        left_panel = ttk.Frame(self.root, padding=(10, 0, 5, 10))
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

        self.image_listbox = tk.Listbox(list_frame, activestyle="dotbox")
        self.image_listbox.grid(row=0, column=0, sticky="nsew")
        self.image_listbox.bind("<<ListboxSelect>>", self.on_image_selected)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.image_listbox.configure(yscrollcommand=scrollbar.set)

        right_panel = ttk.Frame(self.root, padding=(5, 0, 10, 10))
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(right_panel, highlightthickness=0)
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
        self._bind_mousewheel(self.right_canvas)

        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_frame.rowconfigure(4, weight=1)

        ttk.Label(preview_frame, text="Histograma original").grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(preview_frame, text="Imagen original").grid(row=0, column=1, sticky="w", pady=(0, 5))

        self.histogram_container = tk.Frame(preview_frame, bd=1, relief="solid")
        self.histogram_container.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.histogram_container.columnconfigure(0, weight=1)
        self.histogram_container.rowconfigure(0, weight=1)

        self.original_container = tk.Frame(preview_frame, bd=1, relief="solid")
        self.original_container.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        self.original_container.columnconfigure(0, weight=1)
        self.original_container.rowconfigure(0, weight=1)

        ttk.Label(preview_frame, text="Histograma procesado").grid(row=2, column=0, sticky="w", pady=(0, 5))
        ttk.Label(preview_frame, text="Imagen procesada").grid(row=2, column=1, sticky="w", pady=(0, 5))

        self.processed_histogram_container = tk.Frame(preview_frame, bd=1, relief="solid")
        self.processed_histogram_container.grid(row=3, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        self.processed_histogram_container.columnconfigure(0, weight=1)
        self.processed_histogram_container.rowconfigure(0, weight=1)

        self.processed_container = tk.Frame(preview_frame, bd=1, relief="solid")
        self.processed_container.grid(row=3, column=1, sticky="nsew", pady=(0, 10))
        self.processed_container.columnconfigure(0, weight=1)
        self.processed_container.rowconfigure(0, weight=1)

        metrics_frame = ttk.LabelFrame(right_panel, text="Métricas comparativas", padding=10)
        metrics_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        metrics_frame.columnconfigure(1, weight=1)

        ttk.Label(metrics_frame, text="Desviación estándar original:").grid(row=0, column=0, sticky="w")
        self.original_std_value_label = ttk.Label(metrics_frame, text="-")
        self.original_std_value_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(metrics_frame, text="Desviación estándar procesada:").grid(row=1, column=0, sticky="w")
        self.processed_std_value_label = ttk.Label(metrics_frame, text="-")
        self.processed_std_value_label.grid(row=1, column=1, sticky="w", padx=(10, 0))

        ttk.Label(metrics_frame, text="AMBE:").grid(row=2, column=0, sticky="w")
        self.ambe_value_label = ttk.Label(metrics_frame, text="-")
        self.ambe_value_label.grid(row=2, column=1, sticky="w", padx=(10, 0))

        ttk.Label(metrics_frame, text="PSNR:").grid(row=3, column=0, sticky="w")
        self.psnr_value_label = ttk.Label(metrics_frame, text="-")
        self.psnr_value_label.grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.evaluation_label = ttk.Label(right_panel, text="Evaluación: -", wraplength=500, justify="left")
        self.evaluation_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.status_label = ttk.Label(self.root, text="Listo", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self._show_placeholder(self.original_container, "Selecciona una carpeta para comenzar")
        self._show_placeholder(self.processed_container, "La imagen procesada aparecerá aquí")
        self._show_placeholder(self.histogram_container, "Selecciona una imagen para ver su histograma")
        self._show_placeholder(self.processed_histogram_container, "El histograma procesado aparecerá aquí")

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

        self.image_paths = list_image_files(folder_path)
        self._refresh_image_list()

        if self.image_paths:
            self.status_label.configure(text=f"{len(self.image_paths)} imagen(es) detectada(s)")
            self.image_listbox.selection_set(0)
            self.image_listbox.event_generate("<<ListboxSelect>>")
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

    def _show_placeholder(self, container: tk.Widget, text: str) -> None:
        """Show a simple text placeholder in the preview area."""
        self._clear_preview(container)
        for child in container.winfo_children():
            child.destroy()

        label = ttk.Label(container, text=text, anchor="center", justify="center")
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
        axes.set_title(title)
        axes.axis("off")
        figure.tight_layout()

        self.original_canvas = FigureCanvasTkAgg(figure, master=self.original_container)
        widget = self.original_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.original_canvas.draw()

    def _show_processed_image(self, image_gray: np.ndarray, title: str) -> None:
        """Render the processed grayscale image."""
        self._clear_preview(self.processed_container)
        for child in self.processed_container.winfo_children():
            child.destroy()

        figure = Figure(figsize=(4.6, 4.6), dpi=100)
        axes = figure.add_subplot(111)
        axes.imshow(image_gray, cmap="gray")
        axes.set_title(title)
        axes.axis("off")
        figure.tight_layout()

        self.processed_canvas = FigureCanvasTkAgg(figure, master=self.processed_container)
        widget = self.processed_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.processed_canvas.draw()

    def _show_histogram(self, image_bgr: np.ndarray, title: str, container: tk.Widget) -> None:
        """Render the grayscale histogram for the selected image."""
        self._clear_preview(container)
        for child in container.winfo_children():
            child.destroy()

        histogram = calculate_grayscale_histogram(image_bgr)

        figure = Figure(figsize=(4.6, 2.6), dpi=100)
        axes = figure.add_subplot(111)
        axes.plot(histogram, color="black", linewidth=1)
        axes.set_xlim([0, 255])
        axes.set_title(f"Histograma en escala de grises: {title}")
        axes.set_xlabel("Intensidad")
        axes.set_ylabel("Frecuencia")
        axes.grid(alpha=0.2)
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=container)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        canvas.draw()

        if container is self.histogram_container:
            self.histogram_canvas = canvas
        elif container is self.processed_histogram_container:
            self.processed_histogram_canvas = canvas

    def _bind_mousewheel(self, canvas: tk.Canvas) -> None:
        """Enable mouse wheel scrolling on the right panel."""
        def _on_mousewheel(event: tk.Event) -> str:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

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
        if self.selected_image is not None and self.selected_image_name is not None:
            self.status_label.configure(
                text=f"Algoritmo cambiado a {self.algorithm_var.get()}. Pulsa Procesar para actualizar."
            )

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
        self.status_label.configure(
            text=f"Procesado con {algorithm_name}: {self.selected_image_name}"
        )

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
