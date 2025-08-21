import reflex as rx

from rxconfig import styles_path

from .components import navbar


@rx.page("/")
def index():
    return rx.vstack(
        navbar(),
        rx.center(
            rx.text("Hello, Reflex!"),
            border_radius="15px",
            border_width="thick",
            width="50%",
            margin_top="20vh",
            margin="auto",
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
