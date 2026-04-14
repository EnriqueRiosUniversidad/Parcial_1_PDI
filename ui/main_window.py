"""Minimal Tkinter user interface for browsing medical images."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.image_loader import list_image_files, load_image


class ImageApp:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Medical Contrast Explorer")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)

        self.current_folder: Path | None = None
        self.image_paths: list[Path] = []
        self.canvas: FigureCanvasTkAgg | None = None

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
        right_panel.rowconfigure(1, weight=1)

        ttk.Label(right_panel, text="Vista previa").grid(row=0, column=0, sticky="w", pady=(0, 5))

        self.preview_container = tk.Frame(right_panel, bd=1, relief="solid")
        self.preview_container.grid(row=1, column=0, sticky="nsew")
        self.preview_container.columnconfigure(0, weight=1)
        self.preview_container.rowconfigure(0, weight=1)

        self.status_label = ttk.Label(self.root, text="Listo", anchor="w")
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self._show_placeholder("Selecciona una carpeta para comenzar")

    def run(self) -> None:
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def select_folder(self) -> None:
        """Open a folder picker and load supported images."""
        selected_folder = filedialog.askdirectory(title="Selecciona la carpeta con imágenes JPG")
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
            self._show_placeholder("No se pudo cargar la imagen")
            return

        self._show_image(image, image_path.name)
        self.status_label.configure(text=str(image_path))

    def _clear_preview(self) -> None:
        """Remove the current preview widget if it exists."""
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

    def _show_placeholder(self, text: str) -> None:
        """Show a simple text placeholder in the preview area."""
        self._clear_preview()
        for child in self.preview_container.winfo_children():
            child.destroy()

        label = ttk.Label(self.preview_container, text=text, anchor="center", justify="center")
        label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def _show_image(self, image_bgr, title: str) -> None:
        """Render an image on the embedded Matplotlib canvas."""
        self._clear_preview()
        for child in self.preview_container.winfo_children():
            child.destroy()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        figure = Figure(figsize=(6, 6), dpi=100)
        axes = figure.add_subplot(111)
        axes.imshow(image_rgb, cmap="gray" if len(image_rgb.shape) == 2 else None)
        axes.set_title(title)
        axes.axis("off")
        figure.tight_layout()

        self.canvas = FigureCanvasTkAgg(figure, master=self.preview_container)
        widget = self.canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        self.canvas.draw()
