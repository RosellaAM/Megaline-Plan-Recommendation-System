# Megaline Plan Recommendation System - Smart/Ultra Mobile Plan Predictive Model

[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-blueviolet)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Megaline Plan Recommendation System** is a machine learning solution for Megaline company that automatically recommends the optimal mobile plan (Smart or Ultra) to customers based on their historical usage behavior. The project addresses the challenge of migrating customers from legacy plans to new plans through a binary classification model achieving over 75% accuracy.

## 🚀 Results
The final **Random Forest** model demonstrated:
- Over 75% accuracy on test data.
- Strong generalization capability for new customers.
- Optimal balance between recall and precision.
- Validation through sanity testing against simple baseline.

## 💼 Business Impact
- **Churn Reduction**: Improved customer satisfaction with appropriate plan recommendations.
- **Revenue Optimization**: Effective migration to premium plans when relevant.
- **Automation**: Scalable system for real-time recommendations.
- **Data-Driven Decisions**: Strategy based on actual usage patterns.

## 🎯 Core Skills
* Exploratory Data Analysis (EDA): Data cleaning, distribution analysis, correlation identification, and user behavior pattern recognition.
* Data Preprocessing: Handling structured data, stratified splitting to maintain class proportions.
* Feature Engineering: Analysis of relevant variables for mobile plan prediction (calls, minutes, messages, data usage).
* Binary Classification: Implementation of models to predict between two categories (Smart vs Ultra).
* Algorithm Comparison: Evaluation of Random Forest, Decision Tree, and Logistic Regression.
* Hyperparameter Optimization: Fine-tuning models to maximize accuracy and generalization.
* Statistical Visualizations: Using Matplotlib and Seaborn for distribution plots and correlation analysis.
* Results Analysis: Interpretation of classification metrics (accuracy, recall, F1-score).
* Sanity Testing: Validation against baseline models to ensure real business value.

## 🛠️ Tech Stack
* **Machine Learning** → Scikit-learn
* **Backend** → Python 3.8+, Pandas, NumPy
* **Visualization** → Matplotlib, Seaborn
* **Development** → Jupyter Notebooks

## Local Execution
1. Clone the repository:

git clone https://github.com/RosellaAM/Megaline-Plan-Recommendation.git

2. Install dependencies:

pip install -r requirements.txt

3. Run analysis:

  jupyter notebook notebooks/megaline_plan_recommendation_model.ipynb

