import click
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.align import Align
import pyfiglet
import colorama

import torch as th

from pathlib import Path

import asyncio

from main import train_model
from utils import console, save_corpus, get_data, create_version_db

colorama.init(autoreset=True)

@click.group()
@click.option('--nano', '-n', is_flag=True, help='Muestra la saludo de nano')
def cli(nano):
    """Herramienta de entrenamiento para el modelo de predicción de texto."""
    if nano:
        console.print(
            f"[bold blue]{pyfiglet.figlet_format('Nano CLI', font='isometric3', justify='center')}[/]"
        )
    pass

@cli.command()
def init():
    """Inicializar la base de datos."""
    result = create_version_db()
    if result:
        console.print("[bold green]Base de datos inicializada[/bold green]")
    else:
        console.print("[bold red]Error al inicializar la base de datos[/bold red]")

@cli.command()
@click.option('--webs-urls', '-w', multiple=True, help='URLs de las webs a scrapear')
@click.option('--tokens', '-t', default=15, help='Número de tokens a scrapear')
def collect(webs_urls, tokens):
    """Recolectar datos de las webs."""
    console.print(f"[bold green]Recolectando datos de las siguientes webs:[/bold green] {webs_urls}")
    df = asyncio.run(get_data(tokens=tokens, webs_urls=webs_urls))
    if df is None:
        console.print("[bold red]Error al recolectar datos[/bold red]")
        return
    save_corpus(df, tokens)
    console.print("[bold green]Datos recolectados y guardados[/bold green]")

@cli.command()
@click.option('--epochs', '-e', default=10, help='Número de épocas de entrenamiento')
@click.option('--batch-size', '-b', default=64, help='Tamaño del lote para entrenamiento')
@click.option('--learning-rate', '-lr', default=0.001, help='Tasa de aprendizaje')
@click.option('--csv-path', '-csp', default='corpus-20250902174622-tokens_40.csv', help='Ruta del corpus')
@click.option('--model-path', '-m', default='', help='Ruta para guardar el modelo entrenado')
@click.option('--hidden-size', '-hs', default=600, help='Tamaño de las capas ocultas')
@click.option('--emb-size', '-es', default=600, help='Tamaño de embedding')
@click.option('--dropout', '-d', default=0.3, help='Tasa de dropout')
def train(epochs, batch_size, learning_rate, csv_path, model_path, hidden_size, emb_size, dropout):
    """Entrenar el modelo de predicción de texto."""
    console.print(f"[bold green]Iniciando entrenamiento con los siguientes parámetros:[/bold green]")
    console.print(f"Épocas: {epochs}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Learning rate: {learning_rate}")
    console.print(f"Csv Data Set: {csv_path}")
    console.print(f"Ruta del modelo: {model_path}")
    console.print(f"Hidden size: {hidden_size}")
    console.print(f"Embedding size: {emb_size}")
    console.print(f"Dropout: {dropout}")
    
    asyncio.run(train_model(epochs, batch_size, learning_rate, csv_path, model_path, hidden_size, emb_size, dropout))

@cli.command()
@click.option('--model-path', '-m', default='./model', help='Ruta del modelo entrenado')
@click.option('--input-text', '-i', required=True, help='Texto de entrada para generar predicción')
@click.option('--max-length', '-ml', default=50, help='Longitud máxima de la generación')
def predict(model_path, input_text, max_length):
    """Generar predicciones con el modelo entrenado."""
    console.print(f"[bold green]Generando predicción para:[/bold green] {input_text}")
    console.print(f"Usando modelo en: {model_path}")
    console.print(f"Longitud máxima: {max_length}")

if __name__ == '__main__':
    cli()