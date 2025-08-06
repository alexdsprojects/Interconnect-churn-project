# src/predict.py

import pandas as pd
import joblib

# Rutas
MODEL_PATH = 'outputs/models/lgbm_churn_model.pkl'
# Usamos los datos procesados como si fueran "nuevos datos"
NEW_DATA_PATH = 'data/processed/churn_data_processed.csv' 
PREDICTIONS_OUTPUT_PATH = 'outputs/predictions.csv'

def make_predictions():
    """
    Carga el modelo entrenado y hace predicciones sobre nuevos datos.
    """
    print("Cargando modelo y haciendo predicciones...")

    # Cargar el modelo
    model = joblib.load(MODEL_PATH)
    
    # Cargar nuevos datos (sin la columna 'Churn')
    # En un caso real, estos datos no tendrían la columna objetivo
    new_data = pd.read_csv(NEW_DATA_PATH).drop('Churn', axis=1)

    # Realizar predicciones
    predictions_proba = model.predict_proba(new_data)[:, 1] # Probabilidad de Churn (clase 1)
    
    # Crear un DataFrame con los resultados
    results_df = pd.DataFrame({
        'Churn_Probability': predictions_proba
    })
    
    # (Opcional) Unir las predicciones a los datos originales para tener contexto
    # Para esto, necesitarías el customerID, que quitamos. 
    # Una mejor práctica sería mantenerlo hasta el final del preproceso.
    
    results_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"Predicciones guardadas en: {PREDICTIONS_OUTPUT_PATH}")

if __name__ == '__main__':
    make_predictions()