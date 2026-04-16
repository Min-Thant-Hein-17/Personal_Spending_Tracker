# 💰 Financial Health Score Predictor — ML Web App

## Project Overview
This Streamlit web application uses machine learning to predict a person's **Financial Health Score** 
based on their income, expenses, savings, and demographic information.

**Algorithm: Regression** — because the target variable (`financial_health_score`) is a continuous 
numerical value (0–100), not a category.

## Dataset
- File: `personal_spending_dataset.csv`
- Rows: ~10,000  
- Target: `financial_health_score`

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```
The app will open at: http://localhost:8501

## App Pages
| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and workflow |
| 📊 Data Overview | Dataset exploration and visualizations |
| 🔧 Preprocessing | Encoding, outlier handling, feature engineering |
| 🤖 Model Training | 4 regression models with hyperparameter info |
| 📈 Evaluation | RMSE, MAE, R², feature importance charts |
| 🔮 Predict | Interactive form to predict your own score |

## Models Used
- **Linear Regression** (baseline)
- **Ridge Regression** (L2 regularization)
- **Random Forest Regressor** (n_estimators=100, max_depth=10)
- **Gradient Boosting Regressor** (n_estimators=100, learning_rate=0.1)

## Preprocessing Steps
1. Missing value check
2. Binary encoding (Yes/No → 1/0)
3. Label encoding (categorical columns)
4. Outlier capping (debt at 99th percentile)
5. Feature engineering (total_expenses, expense_to_income_ratio)
6. StandardScaler (for linear models)
7. 80/20 train-test split

## Author
Mid-Term Project — ML Web App Development
