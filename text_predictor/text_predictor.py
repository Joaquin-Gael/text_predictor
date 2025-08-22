import reflex as rx

from rxconfig import styles_path

from .components import navbar, chat, input_prompt
from .api import ChatState


@rx.page("/")
def index():
    return rx.vstack(
        navbar("Text Predictor"),
        rx.center(
            rx.box(
                chat(),
                input_prompt(),
                class_name="main p-6 bg-white border border-gray-200 rounded-lg shadow-sm dark:bg-gray-800 dark:border-gray-700 w-[100vw] sm:w-[50vw]",
            ),
            width="100vw",
        )
    )


app = rx.App(
    stylesheets=[styles_path.absolute().as_posix()],
    theme=rx.theme(
        appearance="dark",
        accent_color="teal",
        radius="large",
        has_background=True,
    ),
)
