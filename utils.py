from typing import Optional, List, Dict, Any, Tuple, Union

import pandas as pd
import rich
import tiktoken

import requests as rq
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from fake_headers import Headers

import re
import unicodedata
import time
import random

import numpy as np

from tbparse import SummaryReader

from datetime import datetime
from pathlib import Path

import math

import polars as pl

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


# Configuración de sesión con reintentos
def create_session():
    """Crea una nueva sesión con configuración de reintentos"""
    new_session = rq.Session()
    retry = Retry(
        total=5,
        backoff_factor=2,  # Aumentado para esperar más tiempo entre reintentos
        status_forcelist=[429, 500, 502, 503, 504],  # Añadido 429 (Too Many Requests)
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        respect_retry_after_header=True  # Respetar el header Retry-After
    )
    new_session.mount("https://", HTTPAdapter(max_retries=retry))
    new_session.mount("http://", HTTPAdapter(max_retries=retry))
    new_session.timeout = 15  # Aumentado el timeout
    return new_session

# Crear un pool de sesiones para rotar
session_pool = [create_session() for _ in range(12)]
current_session_index = 0

# Generador de headers aleatorios
header_generator = Headers(headers=True)

# Lista de referrers para rotación
referrers = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://www.yahoo.com/",
    "https://duckduckgo.com/",
    "https://www.reddit.com/"
]

# Lista de idiomas para rotación
languages = [
    "es-ES,es;q=0.9,en;q=0.8",
    "es-MX,es;q=0.9,en;q=0.8",
    "es-AR,es;q=0.9,en;q=0.7",
    "es;q=0.9,en-US;q=0.8,en;q=0.7"
]

def get_html(url: str) -> str | None:
    """Obtiene el HTML de una URL con rotación de headers y sesiones para evitar baneos"""
    global current_session_index
    
    session = session_pool[current_session_index]
    current_session_index = (current_session_index + 1) % len(session_pool)
    
    headers = header_generator.generate()
    
    # Añadir referrers aleatorios para parecer más natural
    headers['Referer'] = random.choice(referrers)
    
    # Añadir Accept-Language aleatorio (enfocado en español)
    headers['Accept-Language'] = random.choice(languages)
    
    # Añadir otros headers para parecer más humano
    headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    headers['Connection'] = 'keep-alive'
    headers['DNT'] = '1'  # Do Not Track
    
    try:
        # Añadir un retraso aleatorio entre peticiones (entre 1 y 5 segundos)
        wait_time = random.uniform(1, 5)
        #console.print(f"[bold blue]Esperando {wait_time:.2f} segundos antes de solicitar {url}...[/bold blue]")
        time.sleep(wait_time)
        
        # Realizar la petición con los headers personalizados
        response = session.get(url, headers=headers, timeout=(10, 30))  # (connect timeout, read timeout)
        response.raise_for_status()
        
        # Si recibimos un código 429 (Too Many Requests), esperar más tiempo
        if response.status_code == 429:
            wait_time = int(response.headers.get('Retry-After', 60))
            console.print(f"[bold yellow]Recibido 429, esperando {wait_time} segundos...[/bold yellow]")
            time.sleep(wait_time)
            return get_html(url)  # Reintentar después de esperar
        
        html = response.text
        return html
    except rq.exceptions.RequestException as e:
        console.print(f"[bold red]Error al obtener {url}: {str(e)}[/bold red]")
        console.print_exception(show_locals=True)
        
        # Si es un error de conexión, esperar más tiempo antes de reintentar
        if isinstance(e, (rq.exceptions.ConnectionError, rq.exceptions.Timeout)):
            wait_time = random.uniform(10, 30)
            console.print(f"[bold yellow]Error de conexión, esperando {wait_time:.2f} segundos...[/bold yellow]")
            time.sleep(wait_time)
        
        return None
    
    except:
        console.print_exception(show_locals=True)

# Lista de proxies (ejemplo - deberías usar proxies reales)
proxies = [
    # Formato: "http://usuario:contraseña@ip:puerto"
    # Estos son ejemplos y no funcionarán, debes reemplazarlos con proxies reales
    None,  # Sin proxy (conexión directa)
    # "http://user:pass@123.45.67.89:8080",
    # "http://user:pass@98.76.54.32:3128",
]

# Función para obtener un proxy aleatorio
def get_random_proxy():
    return random.choice(proxies)

# Función para aplicar un proxy a una sesión
def apply_proxy_to_session(session, proxy):
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy
        }
    return session

def get_soup_from_url(urls:list[str] | None = None, soup:list[str] = [], extensions:list[str] | None = None) -> list[str]:
    if urls and extensions:
        url = urls[0]
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Descargando datos..."),
            BarColumn(),
            TextColumn("URL: {task.fields[url]}"),
            TimeElapsedColumn(),
            TextColumn("Status: {task.fields[status]}"),
            TextColumn("[red]TimeOut: {task.fields[timeout]}")
        ) as progress:
            task = progress.add_task("Descargando...", total=len(urls), url="None", status="None", timeout=0.0)
            for i, ext in enumerate(extensions):
                html = get_html(url+ext)
                if html:
                    soup.extend(BeautifulSoup(html, features="lxml").get_text().splitlines())
                    if i < len(extensions) - 1:
                        wait_time = random.uniform(3, 8)
                        progress.update(task, advance=1, url=url+ext, status="Success", timeout=wait_time)
                        time.sleep(wait_time)
                    else:
                        progress.update(task, advance=1, url=url+ext, status="Success", timeout=0.0)
                else:
                    progress.update(task, advance=1, url=url+ext, status="Failed", timeout=0.0)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Descargando datos..."),
            BarColumn(),
            TextColumn("URL: {task.fields[url]}"),
            TimeElapsedColumn(),
            TextColumn("Status: {task.fields[status]}"),
            TextColumn("[red]TimeOut: {task.fields[timeout]}")
        ) as progress:
            task = progress.add_task("Descargando...", total=len(urls), url="None", status="None", timeout=0.0)
            for i, url in enumerate(urls):
                html = get_html(url)
                if html:
                    soup.extend(BeautifulSoup(html, features="lxml").get_text().splitlines())
                    if i < len(urls) - 1:
                        wait_time = random.uniform(3, 8)
                        progress.update(task, advance=1, url=url, status="Success", timeout=wait_time)
                        time.sleep(wait_time)
                    else:
                        progress.update(task, advance=1, url=url, status="Success", timeout=0.0)
                else:
                    progress.update(task, advance=1, url=url, status="Failed", timeout=0.0)

    return soup


# Función para obtener datos con técnicas anti-baneo
def get_data(tokens: int = 10, webs_urls: Optional[List[str]] = None) -> pl.DataFrame | None:
    try:
        # Crear un timestamp para guardar los datos descargados
        #timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        wiki_urls = [
            "https://es.wikipedia.org/wiki/Argentina",
            "https://es.wikipedia.org/wiki/Python",
            "https://es.wikipedia.org/wiki/Albert_Einstein",
            "https://es.wikipedia.org/wiki/Germán_Garmendia",
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
        
        if webs_urls:
            wiki_urls.extend(webs_urls)
        
        random.shuffle(wiki_urls)

        soup = get_soup_from_url(wiki_urls)
        
        python_url_base = "https://docs.python.org/es/3/tutorial/"
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
        
        random.shuffle(python_sub_html)
        
        soup = get_soup_from_url([python_url_base], soup, python_sub_html)

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
            "pred0": [],
        }

        for i in range(tokens):
            couple_words[f"token{i}"] = []

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
                    "pred0": [],
                }

                for i in range(tokens):
                    couple[f"token{i}"] = []

                for i in range(len(encoded)-tokens):

                    for j in range(tokens):

                        couple[f"token{j}"].append(encoded[i+j])
                        
                        # Eliminado: if j == len(tokens): ... y encoded("<end>")

                    # Etiqueta: siguiente token inmediato tras la ventana
                    pred = encoded[i+tokens] if i+tokens < len(encoded) else 0

                    couple["pred0"].append(int(pred))


                for i in range(tokens):
                    couple_words[f"token{i}"].extend(couple[f"token{i}"])

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


def gen_graphic(df: pl.DataFrame, n: int, title: str):
    MAX_POINTS = 10000
    WEBGL_THRESHOLD = 3000
    MARKER_THRESHOLD = 2000
    PATIENT_EARLY_STOP = 5      # para detectar posible early stopping (val sube PATIENT_EARLY_STOP épocas seguidas)
    REL_OVERFIT_THRESH = 0.05  # umbral relativo para marcar "OVERFIT" si Val > Train por >5% de la escala

    try:
        df["Delta"] = df["Loss Test"] - df["Loss Train"]

        max_scale = max(df["Loss Train"].max(), df["Loss Test"].max(), 1.0)

        global_min_train_idx = int(df["Loss Train"].idxmin())
        global_min_val_idx = int(df["Loss Test"].idxmin())

        window = 5 if n >= 5 else 1
        df["Train_MA"] = df["Loss Train"].rolling(window=window, center=True, min_periods=1).mean()
        df["Val_MA"] = df["Loss Test"].rolling(window=window, center=True, min_periods=1).mean()

        def is_local_min(series, idx):
            if idx == 0 or idx == len(series)-1:
                return False
            return (series.iloc[idx] < series.iloc[idx-1]) and (series.iloc[idx] < series.iloc[idx+1])


        flags = []
        for i, row in df.iterrows():
            f = []
            if i == global_min_train_idx:
                f.append("GLOBAL_MIN_T")
            if i == global_min_val_idx:
                f.append("GLOBAL_MIN_V")
            if is_local_min(df["Loss Test"], i):
                f.append("LOCAL_MIN_V")
            if df.at[i, "Delta"] > REL_OVERFIT_THRESH * max_scale:
                f.append("OVERFIT")
            if df.at[i, "Loss Test"] > df.at[i, "Loss Train"]:
                f.append("CROSS")
            flags.append(", ".join(f) if f else "-")
        df["Flags"] = flags

        early_stop_epoch = None
        for i in range(0, n - PATIENT_EARLY_STOP):
            window_vals = df["Loss Test"].iloc[i:i+PATIENT_EARLY_STOP+1].values
            # si hay un segmento en el que hay un incremento constante (o mayor promedio)
            if np.all(np.diff(window_vals) > 0):
                early_stop_epoch = int(df["Epoch"].iloc[i+PATIENT_EARLY_STOP])  # marca la época donde ya se confirman las subidas
                break

        summary = {
            "N_epochs": n,
            "Best Test Epoch": int(df.loc[df["Loss Test"].idxmin(), "Epoch"]),
            "Best Test": float(df["Loss Test"].min()),
            "Best Train Epoch": int(df.loc[df["Loss Train"].idxmin(), "Epoch"]),
            "Best Train": float(df["Loss Train"].min()),
            "Test mean (std)": f"{df['Loss Test'].mean():.6f} (±{df['Loss Test'].std():.6f})",
            "Train mean (std)": f"{df['Loss Train'].mean():.6f} (±{df['Loss Train'].std():.6f})",
            "Test 25/50/75%": f"{df['Loss Test'].quantile(0.25):.6f}/{df['Loss Test'].quantile(0.5):.6f}/{df['Loss Test'].quantile(0.75):.6f}",
            "Detected early_stop epoch": early_stop_epoch if early_stop_epoch is not None else "-",
            "Num CROSS (Test>Train)": int((df["Loss Test"] > df["Loss Train"]).sum()),
            "Num OVERFIT flags": int(df["Flags"].str.contains("OVERFIT").sum())
        }

        downsample_step = max(1, math.ceil(n / MAX_POINTS))
        if downsample_step > 1:
            df_ds = df.iloc[::downsample_step].reset_index(drop=True)
        else:
            df_ds = df.copy()

        use_webgl = n >= WEBGL_THRESHOLD
        use_markers = n <= MARKER_THRESHOLD
        Scatter = go.Scattergl if use_webgl else go.Scatter

        customdata = np.stack([
            df_ds["Train_MA"].values,
            df_ds["Val_MA"].values,
            df_ds["Delta"].values,
            df_ds["Flags"].values
        ], axis=1)

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.72, 0.28],
            specs=[[{"type": "xy"}], [{"type": "table"}]]
        )

        train_color = "#1f77b4"
        val_color = "#d62728"

        # Traza training (downsampled)
        fig.add_trace(Scatter(
            x=df_ds["Epoch"],
            y=df_ds["Loss Train"],
            mode=("lines+markers" if use_markers else "lines"),
            name="Train (raw)",
            line=dict(color=train_color, width=1.2),
            marker=dict(size=4) if use_markers else None,
            customdata=customdata,
            hovertemplate=(
                "Epoch: %{x}<br>"
                "Train: %{y:.6f}<br>"
                "Val: %{customdata[2] + %{y:.6f}}<br>"  # not used, left for clarity (we'll rely on Val trace hover)
                "<br><b>Suavizado / Δ / Flags</b><br>"
                "Train MA: %{customdata[0]:.6f}<br>"
                "Val MA: %{customdata[1]:.6f}<br>"
                "Δ (Val-Train): %{customdata[2]:.6f}<br>"
                "Flags: %{customdata[3]}<extra></extra>"
            )
        ), row=1, col=1)

        # Traza validation (downsampled)
        fig.add_trace(Scatter(
            x=df_ds["Epoch"],
            y=df_ds["Loss Test"],
            mode=("lines+markers" if use_markers else "lines"),
            name="Val (raw)",
            line=dict(color=val_color, width=1.2),
            marker=dict(size=4) if use_markers else None,
            customdata=customdata,
            hovertemplate=(
                "Epoch: %{x}<br>"
                "Val: %{y:.6f}<br>"
                "Train: %{customdata[0] - %{customdata[2]:.6f}}<br>"  # aproximación; real train shown in Train trace
                "<br><b>Suavizado / Δ / Flags</b><br>"
                "Train MA: %{customdata[0]:.6f}<br>"
                "Val MA: %{customdata[1]:.6f}<br>"
                "Δ (Val-Train): %{customdata[2]:.6f}<br>"
                "Flags: %{customdata[3]}<extra></extra>"
            )
        ), row=1, col=1)

        # Traza medias móviles (resolución completa para suavizado)
        fig.add_trace(Scatter(
            x=df["Epoch"],
            y=df["Train_MA"],
            mode="lines",
            name=f"Train MA (w={window})",
            line=dict(color=train_color, width=2, dash="dash")
        ), row=1, col=1)
        fig.add_trace(Scatter(
            x=df["Epoch"],
            y=df["Val_MA"],
            mode="lines",
            name=f"Val MA (w={window})",
            line=dict(color=val_color, width=2, dash="dash")
        ), row=1, col=1)

        # Marcar mínimos (global)
        fig.add_trace(go.Scatter(
            x=[int(df.loc[global_min_train_idx, "Epoch"])],
            y=[float(df.loc[global_min_train_idx, "Loss Train"])],
            mode="markers",
            name="Min Train",
            marker=dict(color=train_color, size=10, symbol="diamond")
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[int(df.loc[global_min_val_idx, "Epoch"])],
            y=[float(df.loc[global_min_val_idx, "Loss Test"])],
            mode="markers",
            name="Min Val",
            marker=dict(color=val_color, size=10, symbol="diamond")
        ), row=1, col=1)

        # Si detectamos early_stop, dibujar línea vertical y anotación
        if early_stop_epoch is not None:
            fig.add_vline(x=early_stop_epoch, line_dash="dot", line_color="yellow", row=1, col=1)
            fig.add_annotation(x=early_stop_epoch, y=df["Loss Test"].max(),
                               text=f"Early stop? ép {early_stop_epoch}",
                               showarrow=False, yshift=10, font=dict(color="yellow"), row=1, col=1)

        # --- Tabla resumen abajo ---
        table_header = ["Métrica", "Valor"]
        table_values = [[k for k in summary.keys()], [str(v) for v in summary.values()]]
        fig.add_trace(go.Table(
            header=dict(values=table_header, fill_color="rgba(0,0,0,0.6)", align="left"),
            cells=dict(values=table_values, fill_color="rgba(255,255,255,0.03)", align="left", format=[None, None])
        ), row=2, col=1)

        # --- Layout refinado y hover unified ---
        fig.update_layout(
            title_text=title,
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            #margin=dict(l=60, r=30, t=100, b=40),
            #height=720
        )

        # Ajustes de hoverlabel (más legible)
        fig.update_traces(hoverlabel=dict(align="left", namelength=0, font=dict(size=12)))

        # Mostrar
        return fig

    except Exception as e:
        console.print_exception(show_locals=True)
        print(e)
        return None


def read_scalars_tbparse(logdir: str, tag: Optional[str] = None) -> pd.DataFrame:
    """
    Lee todos los scalars del logdir y devuelve DataFrame con columnas:
    ['tag', 'step', 'value', 'wall_time', 'file', 'run']
    """
    sr = SummaryReader(logdir)
    df = sr.scalars.copy()
    if tag:
        df = df[df['tag'] == tag].copy()
    df = df.sort_values(['run', 'step']).reset_index(drop=True)
    return df

def save_corpus(df: pl.DataFrame, tokens: int):
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
        df.collect()
    else:
        print("No es LazyFrame")
        
    df.write_csv(f"./data/corpus-{datetime.now().strftime('%Y%m%d%H%M%S')}-tokens_{tokens}.csv")