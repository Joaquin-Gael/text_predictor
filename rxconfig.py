import reflex as rx
from pathlib import Path

import torch as th

from main import model
from utils import encode_text, decode_text

tailwind_config = {
    "plugins": ["@tailwindcss/typography"],
    "theme": {
        "extend": {
            "colors": {
                "primary": "#3b82f6",
                "secondary": "#64748b",
            }
        }
    },
}

config = rx.Config(
    app_name="text_predictor",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(tailwind_config),
    ],
)

styles_path = Path(__file__).parent / "assets" / "css" / "main.css"

class ModelState:
    """Estado para el modelo."""
    
    def __init__(self):
        self.model = model
        self.device = "cpu"
        self.tokenizer = encode_text
        self.decode_text = decode_text
        
    def to(self, device: str):
        """Cambiar el dispositivo del modelo."""
        self.device = device
        self.model.to(self.device)
        
        print(f"Model moved to {self.device}")
        print(f"Model device: {next(self.model.parameters()).device}")

    def predict(self, text: str) -> str:
        """Hacer una predicción basada en el texto de entrada."""
        self.model.eval()
        with th.inference_mode():
            input_ids = th.tensor([self.tokenizer(text)], dtype=th.long, device=self.device)
            output_ids = self.model(input_ids)
        return self.decode_text(output_ids[0])
