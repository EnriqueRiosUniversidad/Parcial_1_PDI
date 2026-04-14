"""Minimal Tkinter user interface for browsing medical images."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.image_loader import list_image_files, load_image
from core.histograms import calculate_grayscale_histogram
from core.metrics import calculate_basic_metrics


class ImageApp:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Medical Contrast Explorer")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        self.current_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.original_canvas: FigureCanvasTkAgg | None = None
        self.histogram_canvas: FigureCanvasTkAgg | None = None

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

        self.folder_label = ttk.Label(toolbar, text="Ninguna carpeta seleccionada")
        self.folder_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

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
        right_panel.rowconfigure(1, weight=2)
        right_panel.rowconfigure(2, weight=1)

        ttk.Label(right_panel, text="Imagen original").grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.original_container = tk.Frame(right_panel, bd=1, relief="solid")
        self.original_container.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        self.original_container.columnconfigure(0, weight=1)
        self.original_container.rowconfigure(0, weight=1)

        metrics_frame = ttk.LabelFrame(right_panel, text="Métricas originales", padding=10)
        metrics_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        metrics_frame.columnconfigure(1, weight=1)

        ttk.Label(metrics_frame, text="Media:").grid(row=0, column=0, sticky="w")
        self.mean_value_label = ttk.Label(metrics_frame, text="-")
        self.mean_value_label.grid(row=0, column=1, sticky="w", padx=(10, 0))

        ttk.Label(metrics_frame, text="Desviación estándar:").grid(row=1, column=0, sticky="w")
        self.std_value_label = ttk.Label(metrics_frame, text="-")
        self.std_value_label.grid(row=1, column=1, sticky="w", padx=(10, 0))

        ttk.Label(right_panel, text="Histograma original").grid(row=3, column=0, sticky="w", pady=(0, 5))

        self.histogram_container = tk.Frame(right_panel, bd=1, relief="solid")
        self.histogram_container.grid(row=4, column=0, sticky="nsew")
        self.histogram_container.columnconfigure(0, weight=1)
        self.histogram_container.rowconfigure(0, weight=1)

        self.status_label = ttk.Label(self.root, text="Listo", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self._show_placeholder(self.original_container, "Selecciona una carpeta para comenzar")
        self._show_placeholder(self.histogram_container, "Selecciona una imagen para ver su histograma")

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
            self._show_placeholder(self.histogram_container, "No se pudo cargar la imagen")
            self._set_metrics(None)
            return

        self._show_image(image, image_path.name)
        self._show_histogram(image, image_path.name)
        self._set_metrics(calculate_basic_metrics(image))
        self.status_label.configure(text=str(image_path))

    def _clear_preview(self, container: tk.Widget) -> None:
        """Remove the current preview widget if it exists."""
        if container is self.original_container and self.original_canvas is not None:
            self.original_canvas.get_tk_widget().destroy()
            self.original_canvas = None
        if container is self.histogram_container and self.histogram_canvas is not None:
            self.histogram_canvas.get_tk_widget().destroy()
            self.histogram_canvas = None

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

        figure = Figure(figsize=(6, 6), dpi=100)
        axes = figure.add_subplot(111)
        axes.imshow(image_rgb)
        axes.set_title(title)
        axes.axis("off")
        figure.tight_layout()

        self.original_canvas = FigureCanvasTkAgg(figure, master=self.original_container)
        widget = self.original_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.original_canvas.draw()

    def _show_histogram(self, image_bgr: np.ndarray, title: str) -> None:
        """Render the grayscale histogram for the selected image."""
        self._clear_preview(self.histogram_container)
        for child in self.histogram_container.winfo_children():
            child.destroy()

        histogram = calculate_grayscale_histogram(image_bgr)

        figure = Figure(figsize=(6, 3), dpi=100)
        axes = figure.add_subplot(111)
        axes.plot(histogram, color="black", linewidth=1)
        axes.set_xlim([0, 255])
        axes.set_title(f"Histograma en escala de grises: {title}")
        axes.set_xlabel("Intensidad")
        axes.set_ylabel("Frecuencia")
        axes.grid(alpha=0.2)
        figure.tight_layout()

        self.histogram_canvas = FigureCanvasTkAgg(figure, master=self.histogram_container)
        widget = self.histogram_canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.histogram_canvas.draw()

    def _set_metrics(self, metrics: dict[str, float] | None) -> None:
        """Update the metric labels in the UI."""
        if metrics is None:
            self.mean_value_label.configure(text="-")
            self.std_value_label.configure(text="-")
            return

        self.mean_value_label.configure(text=f"{metrics['media']:.2f}")
        self.std_value_label.configure(text=f"{metrics['desviacion_estandar']:.2f}")
