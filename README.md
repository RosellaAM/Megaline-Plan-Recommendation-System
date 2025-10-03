# Megaline Plan Recommendation System - Smart/Ultra Mobile Plan Predictive Model

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Megaline Plan Recommendation System** is a machine learning solution for Megaline company that automatically recommends the optimal mobile plan (Smart or Ultra) to customers based on their historical usage behavior. The project addresses the challenge of migrating customers from legacy plans to new plans through a binary classification model achieving over 75% accuracy.

## 🚀 Resultados
El modelo final **Random Forest** demostró:
- Precisión superior al 75% en datos de prueba.
- Capacidad de generalización para nuevos clientes.
- Balance óptimo entre recall y precisión.
- Validación mediante prueba de cordura contra baseline simple.

## 💼 Impacto Empresarial
- **Reducción de churn**: Mejor satisfacción del cliente con planes adecuados
- **Optimización de ingresos**: Migración efectiva a planes premium cuando es relevante
- **Automatización**: Sistema escalable para recomendaciones en tiempo real
- **Data-driven decisions**: Decisiones basadas en patrones reales de uso


## 🎯 Habilidades principales
#### Analisis de datos
* Análisis Exploratorio de Datos (EDA): Limpieza de datos, identificación de distribuciones, correlaciones y relaciones entre variables de comportamiento del usuario.
* Preprocesamiento de Datos: Manejo de datos estructurados, división estratificada para mantener proporciones de clases.
* Ingeniería de Features: Análisis de variables relevantes para la predicción de planes móviles (llamadas, minutos, mensajes, datos).

### Modelado predictivo
* Clasificación Binaria: Implementación de modelos para predecir entre dos categorías (Smart vs Ultra).
* Comparación de Algoritmos: Evaluación de Random Forest, Árbol de Decisión y Regresión Logística.
* Optimización de Hiperparámetros: Ajuste fino de modelos para maximizar la precisión y generalización.

### Visualización de datos
* Visualizaciones Estadísticas: Uso de Matplotlib y Seaborn para gráficos de distribución y correlaciones.
* Análisis de Resultados: Interpretación de métricas de clasificación (precisión, recall, F1-score).
* Pruebas de Cordura: Validación contra modelos baseline para asegurar valor real del modelo.

## 🛠️ Stack Tecnológico
* **Frontend** -> Scikit-learn
* **Backend** -> Python 3.8+, Pandas, NumPy
* **Visualización** -> Matplotlib, Seaborn
* **Desarrollo** -> Jupyter Notebooks

## Ejecución Local
1. Clona el repositorio:

git clone https://github.com/RosellaAM/Megaline-Plan-Recommendation.git

2. Instala dependencias:

pip install -r requirements.txt

3. Ejecución de análisis:

  jupyter notebook notebooks/megaline_plan_recommendation_model.ipynb

