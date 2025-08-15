import reflex as rx

config = rx.Config(
    app_name="text_predictor",
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.TailwindV4Plugin(),
    ],
)