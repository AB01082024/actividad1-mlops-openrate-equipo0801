# Actividad 1 - MLOps Open Rate Equipo 0801

## Descripción
Mini flujo MLOps para predicción de Open Rate de notificaciones push.
Proyecto académico - Maestría en Ciencia de Datos 2026.

## Equipo
- Angélica Rocío Becerra Forero - Administradora y Desarrollo principal
- Guillermo Gutiérrez Torres - Desarrollo de modelado y experimentación

## Estructura del proyecto
actividad1-mlops-openrate-equipo0801/
├── src/
│   ├── generate_data.py      # Generación de datos mock
│   ├── prepare_data.py       # Preparación y split de datos
│   ├── train.py              # Modelo Logistic Regression
│   └── train_random_forest.py # Modelo Random Forest
├── data/
│   ├── raw/                  # Datos crudos generados
│   └── processed/            # Datos de entrenamiento y prueba
├── models/                   # Modelos entrenados
├── params.yaml               # Parámetros del proyecto
├── dvc.yaml                  # Pipeline reproducible
└── dvc.lock                  # Estado del pipeline
## Requisitos
- Python 3.11
- Poetry
- DVC
- MLflow

## Instalación
```bash
git clone https://github.com/AB01082024/actividad1-mlops-openrate-equipo0801
cd actividad1-mlops-openrate-equipo0801
poetry install
```

## Ejecutar el pipeline
```bash
poetry run dvc repro
```

## Ver experimentos en MLflow
```bash
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Abrir en el navegador: http://127.0.0.1:5000

## Modelos entrenados
| Modelo | Accuracy | F1 | AUC |
|---|---|---|---|
| Logistic Regression | 0.93 | 0.78 | 0.85 |
| Random Forest | 1.00 | 1.00 | 1.00 |

> **Nota sobre las métricas del Random Forest:** El modelo de Random Forest alcanza un rendimiento perfecto (Accuracy, F1 y AUC = 1.00) porque la variable objetivo (`target_opened`) en los datos sintéticos se genera mediante una regla determinística y sin ruido a partir de dos de las mismas variables usadas como features (`historical_open_rate` y `days_since_last_open`). Al ser una relación exacta y sin aleatoriedad, un modelo basado en árboles logra separar perfectamente las clases. Esto es un efecto propio de la naturaleza sintética del dataset, no un error de fuga de datos (data leakage) en la evaluación, ya que las métricas se calculan correctamente sobre el conjunto de test.