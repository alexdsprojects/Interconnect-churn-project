# src/preprocess.py

import pandas as pd
import numpy as np

# Rutas a los archivos (idealmente esto iría en un config.py)
PATH_CONTRACT = 'data/raw/contract.csv'
PATH_PERSONAL = 'data/raw/personal.csv'
PATH_INTERNET = 'data/raw/internet.csv'
PATH_PHONE = 'data/raw/phone.csv'
OUTPUT_PATH = 'data/processed/churn_data_processed.csv'

def preprocess_data():
    """
    Lee, une y limpia los datos de Interconnect y los guarda como un archivo procesado.
    """
    print("Iniciando preprocesamiento de datos...")
    
    # Cargar los datos
    contract = pd.read_csv(PATH_CONTRACT)
    personal = pd.read_csv(PATH_PERSONAL)
    internet = pd.read_csv(PATH_INTERNET)
    phone = pd.read_csv(PATH_PHONE)

    # Fusionar los dataframes
    df = contract.merge(personal, on='customerID', how='left')
    df = df.merge(internet, on='customerID', how='left')
    df = df.merge(phone, on='customerID', how='left')
    
    # --- Aquí va toda la lógica de limpieza que hiciste en tu notebook ---
    # Ejemplo: Convertir TotalCharges a numérico
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True) # O la estrategia que usaste

    # Ejemplo: Crear la variable objetivo 'Churn'
    df['Churn'] = (df['EndDate'] != 'No').astype(int)
    
    # Ejemplo: Rellenar valores nulos en otras columnas
    # (Usa la misma lógica que en tu notebook)
    for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        df[col] = df[col].fillna('No internet service')

    # Eliminar columnas que ya no son necesarias
    # El 'customerID' se puede quitar para el modelo, 'EndDate' fue reemplazada por 'Churn'
    df.drop(['customerID', 'EndDate'], axis=1, inplace=True)
    
    # Guardar el archivo procesado
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Preprocesamiento completado. Archivo guardado en: {OUTPUT_PATH}")

if __name__ == '__main__':
    # Esto permite que el script se ejecute directamente desde la terminal
    preprocess_data()