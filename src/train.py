# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMClassifier
import joblib  # Para guardar el modelo

# Ruta al archivo procesado
PROCESSED_DATA_PATH = 'data/processed/churn_data_processed.csv'
MODEL_OUTPUT_PATH = 'outputs/models/lgbm_churn_model.pkl'

def train_model():
    """
    Entrena el modelo de clasificación LightGBM y lo guarda.
    """
    print("Iniciando entrenamiento del modelo...")

    # Cargar datos procesados
    df = pd.read_csv(PROCESSED_DATA_PATH)
    
    # Separar características (X) y objetivo (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Dividir datos en entrenamiento y prueba (¡Solo para entrenar!)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Identificar columnas numéricas y categóricas
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=object).columns.tolist()
    
    # --- Definir el pipeline de preprocesamiento (el mismo de tu notebook) ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # --- Definir el pipeline completo con el modelo ---
    # Usa los mejores parámetros que encontraste con GridSearchCV en tu notebook
    best_params = {
        'learning_rate': 0.05,
        'n_estimators': 200,
        'max_depth': 5,
        # ... (añade los otros parámetros que optimizaste)
    }
    
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=42, **best_params))
    ])
    
    # Entrenar el modelo final con todos los datos de entrenamiento
    model_pipeline.fit(X_train, y_train)
    
    # Evaluar el modelo (opcional, pero bueno para confirmar)
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC en el conjunto de prueba: {auc:.4f}")

    # --- ¡Guardar el pipeline del modelo entrenado! ---
    joblib.dump(model_pipeline, MODEL_OUTPUT_PATH)
    print(f"Modelo guardado exitosamente en: {MODEL_OUTPUT_PATH}")

if __name__ == '__main__':
    train_model()