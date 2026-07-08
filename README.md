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
| Random Forest | Por ejecutar | Por ejecutar | Por ejecutar |