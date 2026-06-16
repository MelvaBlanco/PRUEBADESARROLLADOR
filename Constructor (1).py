"""
Constructor.py

Script principal para ejecutar el flujo ETL y análisis del proyecto.
Este archivo corresponde a la versión en Python del notebook Constructor.ipynb.

Requisitos:
    pip install pandas numpy matplotlib openpyxl scikit-learn

Estructura recomendada:
    PRUEBADESARROLLADOR/
    ├── Constructor.py
    ├── contenedor.py
    └── Datos/
        └── cybersecurity_attacks.csv

Uso:
    python Constructor.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from contenedor import CONTENEDOR


def obtener_ruta_datos() -> Path:
    """
    Define la ruta del archivo de entrada.

    Primero busca el archivo dentro de la carpeta local 'Datos'.
    Si no lo encuentra, usa la ruta absoluta original del computador.
    """

    ruta_local = Path("Datos") / "cybersecurity_attacks.csv"

    ruta_absoluta = Path(
        r"C:\Users\JULIAN ROBLES\Desktop\PRUEBADESARROLLADOR\Datos\cybersecurity_attacks.csv"
    )

    if ruta_local.exists():
        return ruta_local

    return ruta_absoluta


def ejecutar_pipeline() -> pd.DataFrame:
    """
    Ejecuta el proceso completo:
    1. Carga de datos.
    2. Análisis inicial.
    3. Análisis de valores nulos.
    4. Normalización.
    5. Análisis descriptivo.
    6. Estadística general.
    7. Respuesta a preguntas de negocio.
    8. Exportación a Excel.

    Returns:
        pd.DataFrame: DataFrame final procesado.
    """

    # Permite importar módulos ubicados en la carpeta superior, si aplica.
    sys.path.append(os.path.abspath(".."))

    # Inicializar contenedor de funciones.
    container = CONTENEDOR()

    # Ruta de archivo.
    path = obtener_ruta_datos()

    print("=" * 70)
    print("INICIO DEL PIPELINE ETL Y ANÁLISIS")
    print("=" * 70)
    print(f"Archivo de entrada: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos en la ruta: {path}\n"
            "Verifique que el archivo exista dentro de la carpeta 'Datos'."
        )

    # 1. Cargar datos.
    df = container.cargar_excel(path)

    # 2. Análisis inicial.
    df = container.analizar(df)

    # 3. Revisión de valores nulos.
    df = container.analizar_nulos(df)

    # 4. Normalización de variables numéricas.
    df = container.normalizar(df)

    # 5. Análisis descriptivo.
    df = container.analisis_descriptivo(df)

    # 6. Estadística general.
    df = container.estadistica(df)

    # 7. Preguntas de negocio.
    df = container.responder_preguntas_negocio(df)

    # 8. Guardar resultado.
    df = container.guardar_excel(df, "cybersecurity_attacks.xlsx")

    print("=" * 70)
    print("PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 70)

    return df


if __name__ == "__main__":
    ejecutar_pipeline()
