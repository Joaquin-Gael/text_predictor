import rich
import tiktoken

import requests as rq
from bs4 import BeautifulSoup


import polars as pl

from rich.progress import track

enc = tiktoken.encoding_for_model("gpt-4o")

console = rich.console.Console()


def get_data() -> pl.DataFrame | None:
    try:
        html_wiki = rq.get("https://es.wikipedia.org/wiki/Argentina").text

        soup = BeautifulSoup(html_wiki, features="lxml")

        list_words = soup.get_text().splitlines()

        couple_words = {
            "word":[],
            "prediction":[],
        }
        for line in track(list_words, description="Processing...", total=len(list_words)):
            couple = {
                "word":[],
                "prediction":[],
            }
            encoded = enc.encode(line)
            for i, word in enumerate(encoded):
                couple["word"].append(word)
                if not i+1 >= len(encoded):
                    couple["prediction"].append(encoded[i+1])
                else:
                    couple["prediction"].append(enc.encode(" ")[0])

            couple_words["word"].extend(couple["word"])
            couple_words["prediction"].extend(couple["prediction"])


        df = pl.DataFrame(couple_words)

        return df
    except Exception as e:
        console.print_exception(show_locals=True)
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