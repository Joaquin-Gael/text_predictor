import reflex as rx
from pathlib import Path

config = rx.Config(
    app_name="text_predictor",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
)

styles_path = Path(__file__).parent / "assets" / "css" / "main.css"
