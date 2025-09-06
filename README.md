# Text Predictor 🚀

¡Bienvenido/a al repositorio de **Nombre del Proyecto**! Este proyecto es una solución innovadora diseñada para Predecir el texto escrito al mismo estilo de GPT en sus comienzos. Aquí encontrarás todo lo necesario para entender, ejecutar y contribuir al desarrollo de esta aplicación.

## Descripción 📖

Este proyecto tiene como objetivo de convertirse en un modelo entrenado desde cero gratuito y de estudio para todos. Está diseñado para evolucionar de un modelo de texto predictivo. Con Torch e información de Wikipedia y más en **futuro**, buscamos ofrecer una experiencia de aprendizaje.

## Características principales ✨

- **Torch**: Es un framework perfecto para hacer desde modelos pequeños a modelos enormes con GPT.
- **Requests-BeautifulSoup**: Las librerias utilizadas para obtener la información gratuita de interner.
- ~~**Rich-cli**: Una CLI echa para que se pueda interactuar y probar con el modelo en **terminal**.~~(En construction)

## Tecnologías utilizadas 🛠️

Hemos utilizado una combinación de tecnologías modernas y robustas para construir este proyecto. A continuación, te mostramos las principales herramientas y lenguajes que hemos empleado:

<div align="center">
  <img src="https://skillicons.dev/icons?i=pytorch,python,git,github,pycharm" />
</div>

- **Modelo**: [pytorch]
- **Dataset**: [requests, beautifulsoup4, polars]
- **Otras herramientas**: [rich, ~~tensorboard~~]

## Instalación 🛠️

Sigue estos pasos para instalar y ejecutar el proyecto en tu máquina local:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/Joaquin-Gael/text_predictor.git
    ```

2. **Crea el entorno virtual e instala las dependencias**:
   ```bash
   poetry env use
   
   poetry check
   
   poetry install
   
   poetry shell
    ```
   
3. **Compilar el modelo**
   ```bash
   py main.py
   ```