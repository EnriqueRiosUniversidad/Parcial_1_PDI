"""Entry point for the image contrast exploration desktop app."""

from ui.main_window import ImageApp


def main() -> None:
    """Start the Tkinter application."""
    app = ImageApp()
    app.run()


if __name__ == "__main__":
    main()
