import reflex as rx

from typing import Callable
from ..api import ChatState

BUTTON_CLASS_NAME = "text-white bg-purple-700 hover:bg-purple-800 focus:outline-none focus:ring-4 focus:ring-purple-300 font-medium rounded-full text-sm px-5 py-2.5 text-center mb-2"
INPUT_CLASS_NAME = "flex-1 bg-slate-800 border-slate-600 text-white placeholder:text-slate-400 focus:border-purple-500 focus:ring-purple-500/20 rounded-lg"

def navbar_link(text: str, url: str) -> rx.Component:
    return rx.link(
        rx.text(text, size="4", weight="medium"), href=url
    )
    
def navbar_button(text: str = "", url: str = "") -> rx.Component:
    return rx.link(
        rx.button(
            text,
            class_name=BUTTON_CLASS_NAME
        ), 
        href=url,
    )
    
def navbar_button_with_action(text: str = "", action: Callable[[], None] = None) -> rx.Component:
    return rx.button(text, on_click=action, class_name=BUTTON_CLASS_NAME)

def navbar(name: str = "") -> rx.Component:
    return rx.box(
        rx.desktop_only(
            rx.hstack(
                rx.hstack(
                    rx.image(
                        src="/logo.jpg",
                        width="2.25em",
                        height="auto",
                        border_radius="25%",
                    ),
                    rx.heading(
                        name, weight="bold"
                    ),
                    align_items="center",
                ),
                rx.hstack(
                    navbar_button("View on GitHub"),
                    justify="center",
                    align_items="center",
                    spacing="5",
                ),
                justify="between",
                align_items="center",
            ),
        ),
        rx.mobile_and_tablet(
            rx.hstack(
                rx.hstack(
                    rx.image(
                        src="/nano.png",
                        width="1em",
                        height="auto",
                        border_radius="25%",
                    ),
                    rx.heading(
                        "Reflex", weight="bold"
                    ),
                    align_items="center",
                ),
                rx.menu.root(
                    rx.menu.trigger(
                        rx.icon("menu", size=30)
                    ),
                    rx.menu.content(
                        rx.menu.item("Home"),
                        rx.menu.item("About"),
                        rx.menu.item("Pricing"),
                        rx.menu.item("Contact"),
                    ),
                    justify="end",
                ),
                justify="between",
                align_items="center",
            ),
        ),
        position="fixed",
        top="0px",
        z_index="5",
        width="100%",
        class_name="navbar",
    )


def _message_bubble(msg: dict) -> rx.Component:
    is_user = (msg.get("role") == "user")
    user_classes = "max-w-[75%] px-3 py-2 rounded-2xl bg-purple-600 text-white ml-auto"
    bot_classes = "max-w-[75%] px-3 py-2 rounded-2xl bg-gray-700 text-white mr-auto"
    return rx.cond(
        is_user,
        rx.hstack(
            rx.box(
                rx.text(msg.get("text", "")),
                class_name=user_classes,
            ),
            rx.avatar(
                src="/img/defaul-avatar.png",
                size="3",
                fallback="U",
            ),
            justify="end",
            width="100%",
        ),
        rx.hstack(
            rx.avatar(
                src="/img/nano.png",
                size="3",
                fallback="N",
            ),
            rx.box(
                rx.text(msg.get("text", "")),
                class_name=bot_classes,
            ),
            justify="start",
            width="100%",
        ),
    )
    

def chat() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.foreach(ChatState.messages, lambda m: _message_bubble(m)),
            class_name="space-y-3 p-4",
        ),
        class_name="chat h-[60vh] w-full sm:w-[47.5vw] overflow-y-auto bg-black/20 border border-white/10 rounded-xl",
    )


def input_prompt() -> rx.Component:
    return rx.box(
        rx.form(
            rx.hstack(
                rx.input(
                    placeholder="Escribe un mensaje...",
                    type="text",
                    value=ChatState.prompt,
                    on_change=ChatState.set_prompt,
                    class_name=INPUT_CLASS_NAME,
                ),
                rx.button("Enviar", class_name=BUTTON_CLASS_NAME),
                class_name="w-full gap-2",
            ),
            on_submit=ChatState.send,
            class_name="w-full",
        ),
        class_name="p-6 border-t border-purple-500/20 bg-slate-800/30"
    )
