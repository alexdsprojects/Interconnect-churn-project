# Proyecto de Predicción de Abandono de Clientes - Interconnect

Este proyecto tiene como objetivo desarrollar un modelo de machine learning para predecir la probabilidad de que un cliente de la empresa de telecomunicaciones Interconnect cancele su servicio.

## Problema de Negocio

Interconnect desea identificar proactivamente a los clientes en riesgo de abandono para poder ofrecerles incentivos y mejorar la retención. Este modelo proporciona un "score" de riesgo para cada cliente.

## Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

- **/data**: Contiene los datos brutos (`raw`) y los datos procesados (`processed`).
- **/notebooks**: Almacena notebooks de Jupyter para análisis exploratorio.
- **/outputs**: Guarda los resultados del proyecto, como el modelo entrenado (`models`) y gráficos (`images`).
- **/src**: Contiene el código fuente modularizado en scripts de Python.
- `requirements.txt`: Lista de dependencias del proyecto.

## Cómo Usar

1.  **Clonar el repositorio:**
    `git clone https://soundcloud.com/detusound`
2.  **Crear y activar un entorno virtual:**
    `python -m venv venv`
    `source venv/bin/activate` (o `.\venv\Scripts\activate` en Windows)
3.  **Instalar dependencias:**
    `pip install -r requirements.txt`
4.  **Ejecutar el pipeline:**
    - Para procesar los datos: `python src/preprocess.py`
    - Para entrenar el modelo: `python src/train.py`
    - Para generar predicciones con nuevos datos: `python src/predict.py`

## Modelo Final y Resultados

El modelo final es un `LGBMClassifier` optimizado. En el conjunto de prueba, alcanzó un **AUC-ROC de 0.8505**, cumpliendo con los requisitos del proyecto.
