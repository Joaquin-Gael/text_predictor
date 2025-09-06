import reflex as rx
from typing import List, Dict
from rxconfig import ModelState, set_device

model_state = ModelState()

set_device("cpu")

class ChatState(rx.State):
    """Estado para un chat multipropósito."""
    prompt: str = ""
    messages: List[Dict[str, str]] = []
    
    def _compute_responce(self, text: str) -> str:
        """Computar la respuesta basada en el texto de entrada."""
        return model_state.predict(text)

    def send(self):
        """Enviar el mensaje actual y generar una respuesta de ejemplo."""
        text = (self.prompt or "").strip()
        if not text:
            return
        # Agregar mensaje del usuario
        self.messages.append({"role": "user", "text": text})
        # Limpiar el prompt
        self.prompt = ""
        # Respuesta simulada (placeholder). Aquí podrás integrar tus herramientas/LLM.
        reply = f"Nano: {self._compute_responce(text)}"
        self.messages.append({"role": "assistant", "text": reply})