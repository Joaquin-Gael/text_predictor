import rich
import tiktoken

import requests as rq
from bs4 import BeautifulSoup

import re
import unicodedata

import polars as pl

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn
)

enc = tiktoken.encoding_for_model("gpt-4o")

console = rich.console.Console()

SPANISH_LETTERS_PATTERN = re.compile(r"[^A-Za-zÁÉÍÓÚÜáéíóúüÑñ\s]+$")
#RE_NON_LATIN = re.compile(r"[^\p{Latin}\s]+")

def is_spanish_word(text: str) -> str | None:
    if not SPANISH_LETTERS_PATTERN.search(text):
        return text
    else:
        return ""

def only_spanish_letters(text: str) -> str | None:
    text = unicodedata.normalize("NFC", text)

    out_chars = []
    for ch in text:
        if ch.isspace():
            out_chars.append(" ")
            continue
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            try:
                name = unicodedata.name(ch)
            except ValueError:
                out_chars.append(" ")
                continue
            if "LATIN" in name:
                out_chars.append(ch)
                continue
        out_chars.append(" ")

    cleaned = "".join(out_chars).strip()

    #cleaned = RE_NON_LATIN.sub(" ", cleaned)

    cleaned = SPANISH_LETTERS_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    word = is_spanish_word(cleaned)

    return word


def get_data() -> pl.DataFrame | None:
    try:
        html_wiki = rq.get("https://es.wikipedia.org/wiki/Argentina").text

        python_sub_html = [
            "",
            "appetite.html",
            "interpreter.html",
            "interpreter.html#invoking-the-interpreter",
            "interpreter.html#argument-passing",
            "interpreter.html#interactive-mode",
            "interpreter.html#the-interpreter-and-its-environment",
            "interpreter.html#source-code-encoding",
            "introduction.html",
            "introduction.html#using-python-as-a-calculator"
        ]

        soup = BeautifulSoup(html_wiki, features="lxml")
        soup = soup.get_text().splitlines()
        for sub_html in python_sub_html:
            html_python_docs_es = rq.get("https://docs.python.org/es/3/tutorial/"+sub_html).text
            soup.extend(BeautifulSoup(html_python_docs_es, features="lxml").get_text().splitlines())

        html_python_wiki = rq.get("https://es.wikipedia.org/wiki/Python").text

        soup.extend(BeautifulSoup(html_python_wiki, features="lxml").get_text().splitlines())

        more_html_data = [
            "https://es.wikipedia.org/wiki/Albert_Einstein",
            "https://es.wikipedia.org/wiki/Germán_Garmendia",
            "https://espanol.lingolia.com/es/ayuda/glosario",
            "https://es.wikipedia.org/wiki/Miguel_de_Cervantes",
            "https://es.wikipedia.org/wiki/Don_Quijote_de_la_Mancha",
            "https://es.wikipedia.org/wiki/Pablo_Neruda",
            "https://es.wikipedia.org/wiki/Las_Meninas",
            "https://es.wikipedia.org/wiki/La_Gioconda",
            "https://es.wikipedia.org/wiki/Johann_Sebastian_Bach",
            "https://es.wikipedia.org/wiki/Mozart",
            "https://es.wikipedia.org/wiki/Leonhard_Euler",
            "https://es.wikipedia.org/wiki/Inteligencia_artificial",
            "https://es.wikipedia.org/wiki/Poder_ejecutivo",
            "https://es.wikipedia.org/wiki/Carne",
            "https://es.wikipedia.org/wiki/Historia_del_pan",
            "https://es.wikipedia.org/wiki/Batalla_de_Hastings",
            "https://es.wikipedia.org/wiki/Batalla_de_Cannas",
            "https://es.wikipedia.org/wiki/Primera_guerra_mundial",
            "https://es.wikipedia.org/wiki/Christopher_Columbus"
        ]

        for more_html in more_html_data:
            html_more_data = rq.get(more_html).text
            soup.extend(BeautifulSoup(html_more_data, features="lxml").get_text().splitlines())

        console.print("Longitud de la sopa de texto: ", len(soup))

        list_words = soup

        spanish_words = []

        for word in list_words:
            word = only_spanish_letters(word)
            if word:
                spanish_words.append(word)

        list_words = spanish_words


        couple_words = {
            "token0": [],
            "token1": [],
            "token2": [],
            "token3": [],
            "token4": [],
            "token5": [],
            "token6": [],
            "token7": [],
            "token8": [],
            #"token9": [],
            #"token10": [],
            "pred0": [],
        }
        with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("Paso: {task.fields[step]}"),
                TimeElapsedColumn()
        ) as progress:

            task = progress.add_task("Procesando líneas...", total=len(list_words), step="Inicializando")

            sequence_buffer, len_buffer = [], 0

            for line in list_words:
                progress.update(task, step="Codificando línea") 
                encoded = enc.encode(line)

                if len(encoded)-9 < 0:
                    len_buffer += 1
                    sequence_buffer.extend(encoded)
                    continue

                if len_buffer > 1:
                    encoded.extend(sequence_buffer)
                    sequence_buffer = []
                couple = {
                    "token0": [],
                    "token1": [],
                    "token2": [],
                    "token3": [],
                    "token4": [],
                    "token5": [],
                    "token6": [],
                    "token7": [],
                    "token8": [],
                    #"token9": [],
                    #"token10": [],
                    "pred0": [],
                }

                for i in range(len(encoded)-9):
                    
                    t0 = encoded[i]
                    t1 = encoded[i+1]
                    t2 = encoded[i+2]
                    t3 = encoded[i+3]
                    t4 = encoded[i+4]
                    t5 = encoded[i+5]
                    t6 = encoded[i+6]
                    t7 = encoded[i+7]
                    t8 = encoded[i+8]
                    #t9 = encoded[i+9]
                    #t10 = encoded[i+10]
                    preds = encoded[i+9] if i+9 < len(encoded) else 0

                    p0 = int(preds)

                    couple["token0"].append(t0)
                    couple["token1"].append(t1)
                    couple["token2"].append(t2)
                    couple["token3"].append(t3)
                    couple["token4"].append(t4)
                    couple["token5"].append(t5)
                    couple["token6"].append(t6)
                    couple["token7"].append(t7)
                    couple["token8"].append(t8)
                    #couple["token9"].append(t9)
                    #couple["token10"].append(t10)
                    couple["pred0"].append(p0)

                couple_words["token0"].extend(couple["token0"])
                couple_words["token1"].extend(couple["token1"])
                couple_words["token2"].extend(couple["token2"])
                couple_words["token3"].extend(couple["token3"])
                couple_words["token4"].extend(couple["token4"])
                couple_words["token5"].extend(couple["token5"])
                couple_words["token6"].extend(couple["token6"])
                couple_words["token7"].extend(couple["token7"])
                couple_words["token8"].extend(couple["token8"])
                #couple_words["token9"].extend(couple["token9"])
                #couple_words["token10"].extend(couple["token10"])
                couple_words["pred0"].extend(couple["pred0"])
                

                progress.advance(task)


                
                
        console.print("Datos procesados correctamente.", style="bold green")
        console.print(f"Total de palabras procesadas: {couple_words['token0'][:10]}")
        console.print(f"Total de predicciones procesadas: {couple_words['pred0'][:10]}")

        df = pl.DataFrame(couple_words)

        return df
    except Exception as e:
        console.print_exception(show_locals=False)
        print(f"Error al obtener los datos: {e}")
        console.print("No se pudo obtener el DataFrame.", style="bold red")
        return None

def encode_text(text: str) -> list[int]:
    return enc.encode(text)

def decode_text(code: list[int]) -> str:
    return enc.decode(code)

if __name__ == "__main__":
    df = get_data()

    print("Longitud del dataframe: ")
    print(len(df))

    # 🔍 Ver las primeras filas (por defecto 5)
    print("📋 Primeras filas:")
    print(df.head(), end="\n\n")

    # 🔍 Información general y estructura
    print("📋 Info del DataFrame:")
    print(df, end="\n\n")

    # 🔍 Columnas y tipos de datos
    print("📋 Columnas y tipos:")
    print(df.dtypes, end="\n\n")

    # 🔍 Descripción estadística
    print("📊 Descripción estadística:")
    print(df.describe(), end="\n\n")

    # 🔍 Número de filas y columnas
    print(f"📏 Filas: {df.height}, Columnas: {df.width}\n")

    # 🔍 Nombres de las columnas
    print("📋 Nombres de las columnas:")
    print(df.columns, end="\n\n")

    # 🔍 Resumen de nulos por columna
    print("🛑 Resumen de valores nulos:")
    print(df.null_count(), end="\n\n")

    # 🔍 Memoria utilizada
    print(f"💾 Memoria utilizada: {df.estimated_size() / (1024 ** 2):.2f} MB\n")

    # 🔍 Ver las últimas filas
    print("📋 Últimas filas:")
    print(df.tail(), end="\n\n")

    # 🔍 Información detallada (si es un DataFrame Lazy)
    if isinstance(df, pl.LazyFrame):
        print("📋 Plan lógico del LazyFrame:")
        print(df.describe(), end="\n\n")