"""Entrypoint: build the Gradio interface and launch the app."""

from app_layer.ui.gradio_app import create_interface

if __name__ == "__main__":
    try:
        app = create_interface()
    except RuntimeError as e:
        print(f"\n[CONFIG ERROR] {e}\n")
        raise SystemExit(1)
    app.launch()
