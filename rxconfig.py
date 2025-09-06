import reflex as rx
from pathlib import Path

import torch as th

from main import model, get_device
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

_current_device = None

def set_device(device_str: str):
    """Mover el modelo al dispositivo indicado y recordar el estado actual.
    Uso: set_device("cuda") o set_device("cpu").
    """
    global _current_device
    device = th.device(device_str)
    model.to(device)
    _current_device = device
    print(f"Model moved to {device}")
    print(f"Model device: {next(model.parameters()).device}")


def current_device():
    """Devolver el dispositivo efectivo actual (del modelo)."""
    if _current_device is not None:
        return _current_device
    # fallback al device del propio modelo si no fue seteado aún
    try:
        return next(model.parameters()).device
    except Exception:
        return th.device("cuda" if th.cuda.is_available() else "cpu")


class ModelState:
    """Estado para el modelo."""
    
    def __init__(self):
        self.model = model
        self.tokenizer = encode_text
        self.decode_text = decode_text
        
    def to(self, device: str):
        """Cambiar el dispositivo del modelo."""
        set_device(device)
        
    def predict(self, text: str) -> str:
        """Hacer una predicción basada en el texto de entrada."""
        self.model.eval()
        with th.inference_mode():
            dev = current_device()
            input_ids = th.tensor([self.tokenizer(text)], dtype=th.long, device=dev)
            output_ids = self.model.generate(input_ids)
            print(f"loggits: {output_ids}")
        return self.decode_text(output_ids)
