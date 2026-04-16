import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Health Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stButton>button {
        background-color: #2563eb;
        color: blue;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("personal_spending_dataset.csv")
    return df

# ─── Preprocessing ───────────────────────────────────────────────────────────
@st.cache_data
def preprocess_data(df):
    data = df.copy()

    # Encode binary columns
    binary_cols = ["investment", "emergency_fund"]
    for col in binary_cols:
        data[col] = data[col].map({"Yes": 1, "No": 0})

    # Label encode categorical columns
    cat_cols = ["gender", "occupation", "city", "income_source",
                "credit_card_usage", "financial_stress"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        le_dict[col] = le

    # Drop rows with missing target
    data = data.dropna(subset=["financial_health_score"])

    # Cap extreme outliers in debt column (99th percentile)
    debt_cap = data["debt"].quantile(0.99)
    data["debt"] = data["debt"].clip(upper=debt_cap)

    # Feature engineering: total expenses
    expense_cols = ["housing_expense", "food_expense", "transport_expense",
                    "entertainment_expense", "shopping_expense", "healthcare_expense"]
    data["total_expenses"] = data[expense_cols].sum(axis=1)
    data["expense_to_income_ratio"] = data["total_expenses"] / (data["monthly_income"] + 1)

    return data, le_dict

# ─── Train Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    data, le_dict = preprocess_data(df)

    feature_cols = [
        "age", "gender", "occupation", "city", "monthly_income",
        "income_source", "savings_rate", "debt", "housing_expense",
        "food_expense", "transport_expense", "entertainment_expense",
        "shopping_expense", "healthcare_expense", "credit_card_usage",
        "investment", "emergency_fund", "financial_stress",
        "total_expenses", "expense_to_income_ratio"
    ]

    X = data[feature_cols]
    y = data["financial_health_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        ),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        if name in ["Linear Regression", "Ridge Regression"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R² Score": round(r2, 4),
            "y_pred": y_pred,
        }
        trained_models[name] = model

    return trained_models, results, scaler, X_test, y_test, feature_cols, le_dict, data


# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/money.png", width=80)
st.sidebar.title("💰 Financial Health ML")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Data Overview", "🔧 Preprocessing", "🤖 Model Training", "📈 Evaluation", "🔮 Predict"],
    index=0
)

# Load data once
df = load_data()
trained_models, results, scaler, X_test, y_test, feature_cols, le_dict, processed_data = train_models(df)

# ─── HOME PAGE ───────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("💰 Personal Financial Health Score Predictor")
    st.markdown("#### A Machine Learning Web Application")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>10,000</h2>
            <p>Records in Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>19</h2>
            <p>Input Features</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>4</h2>
            <p>Models Compared</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🎯 Project Goal")
    st.info("""
    This application predicts a person's **Financial Health Score** (0–100) based on their 
    spending habits, income, savings, and demographics.

    We use **Regression** because the target variable (`financial_health_score`) is a 
    continuous numerical value — not a category. Regression is the correct approach when 
    predicting quantities rather than classes.
    """)

    st.markdown("### 🗺️ App Workflow")
    steps = [
        ("1️⃣ Data Overview", "Explore the raw dataset — shape, types, and distributions"),
        ("2️⃣ Preprocessing", "Handle missing values, encode categories, engineer features"),
        ("3️⃣ Model Training", "Train & compare 4 regression models with hyperparameters"),
        ("4️⃣ Evaluation", "Assess models using RMSE, MAE, and R² metrics"),
        ("5️⃣ Predict", "Enter your own data to get a financial health prediction"),
    ]
    for title, desc in steps:
        st.markdown(f"**{title}** — {desc}")

# ─── DATA OVERVIEW ───────────────────────────────────────────────────────────
elif page == "📊 Data Overview":
    st.title("📊 Data Overview")
    st.markdown("---")

    st.markdown('<div class="section-header">Dataset Shape & Sample</div>', unsafe_allow_html=True)
    st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column": df.dtypes.index,
            "Type": df.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df["Missing Count"] > 0] if missing.sum() > 0
                     else pd.DataFrame({"Note": ["No missing values found ✅"]}),
                     use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Statistical Summary (Numeric Columns)</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

    st.markdown('<div class="section-header">Target Variable Distribution</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["financial_health_score"], bins=40, color="#2563eb", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution of Financial Health Score")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequency")

    # Category columns distribution
    cat_col = st.selectbox("Select categorical column to explore:", 
                           ["gender", "occupation", "city", "financial_stress", "credit_card_usage"])
    value_counts = df[cat_col].value_counts()
    axes[1].bar(value_counts.index, value_counts.values, color="#10b981", edgecolor="white")
    axes[1].set_title(f"Distribution of {cat_col}")
    axes[1].set_xlabel(cat_col)
    axes[1].set_ylabel("Count")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-header">Correlation Heatmap (Numeric Features)</div>', unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=np.number)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax2, annot_kws={"size": 8})
    ax2.set_title("Correlation Heatmap")
    plt.tight_layout()
    st.pyplot(fig2)

# ─── PREPROCESSING ────────────────────────────────────────────────────────────
elif page == "🔧 Preprocessing":
    st.title("🔧 Data Preprocessing")
    st.markdown("---")

    st.markdown("### Step 1 — Handle Missing Values")
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        st.success("✅ No missing values found in the dataset. No imputation needed.")
    else:
        st.warning(f"⚠️ Found {missing_count} missing values. Rows with missing target are dropped.")

    st.markdown("### Step 2 — Encode Categorical Variables")
    st.info("""
    - **Binary columns** (`investment`, `emergency_fund`): Mapped Yes→1, No→0  
    - **Ordinal/nominal columns** (`gender`, `occupation`, `city`, `income_source`, 
      `credit_card_usage`, `financial_stress`): Applied **Label Encoding**
    """)
    enc_preview = processed_data[["gender", "occupation", "city", "credit_card_usage", "financial_stress"]].head(5)
    st.write("**After Encoding (first 5 rows):**")
    st.dataframe(enc_preview, use_container_width=True)

    st.markdown("### Step 3 — Outlier Treatment")
    st.info("The `debt` column has extreme outliers. We capped values at the **99th percentile** to reduce their impact on model training.")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot(df["debt"].dropna(), vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#fca5a5"))
        ax.set_title("Debt — Before Capping")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.boxplot(processed_data["debt"], vert=False, patch_artist=True,
                   boxprops=dict(facecolor="#6ee7b7"))
        ax.set_title("Debt — After Capping (99th pct)")
        st.pyplot(fig)

    st.markdown("### Step 4 — Feature Engineering")
    st.info("""
    Two new features were created to add predictive power:
    - **`total_expenses`** = sum of all 6 expense columns  
    - **`expense_to_income_ratio`** = total_expenses / monthly_income
    """)
    st.dataframe(processed_data[["monthly_income", "total_expenses", "expense_to_income_ratio"]].head(8),
                 use_container_width=True)

    st.markdown("### Step 5 — Feature Scaling")
    st.info("**StandardScaler** was applied for Linear Regression and Ridge Regression models (zero mean, unit variance). Tree-based models (Random Forest, Gradient Boosting) do not require scaling.")

    st.markdown("### Step 6 — Train / Test Split")
    st.info("The dataset was split: **80% training** (8,000 rows) and **20% testing** (2,000 rows) using `random_state=42` for reproducibility.")

    st.markdown("### Final Feature Set Used for Training")
    feat_df = pd.DataFrame({"Feature": feature_cols, "Index": range(len(feature_cols))})
    st.dataframe(feat_df.set_index("Index"), use_container_width=True)

# ─── MODEL TRAINING ──────────────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    st.markdown("---")

    st.markdown("### Why Regression?")
    st.success("""
    The target variable `financial_health_score` is a **continuous number** ranging from 0 to ~60. 
    Since we are predicting a quantity (not a category), **Regression** is the appropriate algorithm family.
    """)

    st.markdown("### Models Trained")

    model_info = {
        "Linear Regression": {
            "desc": "Baseline model — fits a linear relationship between features and target.",
            "params": "Default (no hyperparameters to tune)",
            "pros": "Fast, interpretable",
            "cons": "Assumes linear relationships — may underfit",
        },
        "Ridge Regression": {
            "desc": "Linear regression with L2 regularization to prevent overfitting.",
            "params": "alpha = 1.0 (regularization strength)",
            "pros": "Handles multicollinearity well",
            "cons": "Still assumes linearity",
        },
        "Random Forest": {
            "desc": "Ensemble of decision trees using bagging. Robust and handles non-linearity.",
            "params": "n_estimators=100, max_depth=10, random_state=42",
            "pros": "High accuracy, handles non-linear data, feature importance",
            "cons": "Slower, less interpretable",
        },
        "Gradient Boosting": {
            "desc": "Sequential ensemble that corrects errors of previous trees.",
            "params": "n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42",
            "pros": "Often best performance, flexible",
            "cons": "Slower to train, more hyperparameters",
        },
    }

    for name, info in model_info.items():
        with st.expander(f"📌 {name}"):
            st.write(f"**Description:** {info['desc']}")
            st.write(f"**Hyperparameters:** {info['params']}")
            st.write(f"✅ **Pros:** {info['pros']}")
            st.write(f"⚠️ **Cons:** {info['cons']}")
            r2 = results[name]["R² Score"]
            st.metric("R² Score", f"{r2:.4f}")

    st.markdown("### Training Summary")
    summary_data = {
        "Model": list(results.keys()),
        "RMSE ↓": [results[m]["RMSE"] for m in results],
        "MAE ↓": [results[m]["MAE"] for m in results],
        "R² ↑": [results[m]["R² Score"] for m in results],
    }
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df.set_index("Model"), use_container_width=True)

    best_model = max(results, key=lambda m: results[m]["R² Score"])
    st.success(f"🏆 Best model: **{best_model}** with R² = {results[best_model]['R² Score']:.4f}")

# ─── EVALUATION ───────────────────────────────────────────────────────────────
elif page == "📈 Evaluation":
    st.title("📈 Model Evaluation")
    st.markdown("---")

    st.markdown("### Evaluation Metrics Explained")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**RMSE** (Root Mean Squared Error)  
        Penalizes large errors more. Lower is better.""")
    with col2:
        st.markdown("""**MAE** (Mean Absolute Error)  
        Average absolute difference. Lower is better.""")
    with col3:
        st.markdown("""**R² Score** (Coefficient of Determination)  
        % of variance explained. Closer to 1 is better.""")

    selected_model = st.selectbox("Select model to inspect:", list(results.keys()))
    res = results[selected_model]

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", res["RMSE"])
    c2.metric("MAE", res["MAE"])
    c3.metric("R² Score", res["R² Score"])

    # Actual vs Predicted scatter
    y_pred = res["y_pred"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.3, color="#3b82f6", s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit")
    axes[0].set_xlabel("Actual Score")
    axes[0].set_ylabel("Predicted Score")
    axes[0].set_title(f"Actual vs Predicted — {selected_model}")
    axes[0].legend()

    residuals = np.array(y_test) - y_pred
    axes[1].hist(residuals, bins=40, color="#8b5cf6", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    st.pyplot(fig)

    # Feature importance (Random Forest)
    st.markdown("### Feature Importance (Random Forest)")
    rf_model = trained_models["Random Forest"]
    importance = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance
    }).sort_values("Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues_r(np.linspace(0.2, 0.8, len(feat_imp_df)))
    ax2.barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color=colors)
    ax2.set_xlabel("Importance Score")
    ax2.set_title("Random Forest — Feature Importances")
    ax2.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig2)

    # Model comparison bar chart
    st.markdown("### Model Comparison")
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    metrics = ["RMSE", "MAE", "R² Score"]
    palette = ["#ef4444", "#f97316", "#22c55e"]
    for i, metric in enumerate(metrics):
        vals = [results[m][metric] for m in results]
        axes3[i].bar(list(results.keys()), vals, color=palette[i], edgecolor="white")
        axes3[i].set_title(metric)
        axes3[i].set_xticklabels(list(results.keys()), rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)

# ─── PREDICT ─────────────────────────────────────────────────────────────────
elif page == "🔮 Predict":
    st.title("🔮 Predict Financial Health Score")
    st.markdown("---")
    st.info("Fill in your financial details below. The model will estimate your Financial Health Score (0–100).")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", 18, 70, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            occupation = st.selectbox("Occupation", ["Employee", "Freelancer", "Business", "Student", "Unemployed"])
            city = st.selectbox("City Type", ["Urban", "Suburban", "Rural"])
            income_source = st.selectbox("Income Source", ["Salary", "Freelance", "Business", "Multiple"])

        with col2:
            monthly_income = st.number_input("Monthly Income ($)", 500.0, 15000.0, 2500.0, step=100.0)
            savings_rate = st.slider("Savings Rate (0–1)", 0.0, 0.6, 0.2, step=0.01)
            debt = st.number_input("Total Debt ($)", 0.0, 50000.0, 3000.0, step=500.0)
            credit_card_usage = st.selectbox("Credit Card Usage", ["Low", "Medium", "High"])
            financial_stress = st.selectbox("Financial Stress Level", ["Low", "Medium", "High"])

        with col3:
            housing_expense = st.number_input("Housing Expense ($)", 0.0, 3000.0, 700.0, step=50.0)
            food_expense = st.number_input("Food Expense ($)", 0.0, 1000.0, 250.0, step=10.0)
            transport_expense = st.number_input("Transport Expense ($)", 0.0, 800.0, 150.0, step=10.0)
            entertainment_expense = st.number_input("Entertainment Expense ($)", 0.0, 600.0, 100.0, step=10.0)
            shopping_expense = st.number_input("Shopping Expense ($)", 0.0, 600.0, 120.0, step=10.0)
            healthcare_expense = st.number_input("Healthcare Expense ($)", 0.0, 500.0, 80.0, step=10.0)
            investment = st.selectbox("Has Investments?", ["Yes", "No"])
            emergency_fund = st.selectbox("Has Emergency Fund?", ["Yes", "No"])

        chosen_model = st.selectbox("Choose Prediction Model", list(trained_models.keys()))
        submitted = st.form_submit_button("🔮 Predict My Score")

    if submitted:
        # Encode inputs using the same mappings
        gender_enc = le_dict["gender"].transform([gender])[0]
        occupation_enc = le_dict["occupation"].transform([occupation])[0]
        city_enc = le_dict["city"].transform([city])[0]
        income_source_enc = le_dict["income_source"].transform([income_source])[0]
        credit_enc = le_dict["credit_card_usage"].transform([credit_card_usage])[0]
        stress_enc = le_dict["financial_stress"].transform([financial_stress])[0]
        inv_enc = 1 if investment == "Yes" else 0
        emer_enc = 1 if emergency_fund == "Yes" else 0

        total_expenses = (housing_expense + food_expense + transport_expense +
                          entertainment_expense + shopping_expense + healthcare_expense)
        exp_ratio = total_expenses / (monthly_income + 1)

        input_array = np.array([[
            age, gender_enc, occupation_enc, city_enc, monthly_income,
            income_source_enc, savings_rate, debt, housing_expense,
            food_expense, transport_expense, entertainment_expense,
            shopping_expense, healthcare_expense, credit_enc,
            inv_enc, emer_enc, stress_enc, total_expenses, exp_ratio
        ]])

        model = trained_models[chosen_model]

        if chosen_model in ["Linear Regression", "Ridge Regression"]:
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
        else:
            prediction = model.predict(input_array)[0]

        prediction = max(0, prediction)

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Predicted Financial Health Score", f"{prediction:.2f} / 100")
            if prediction >= 30:
                st.success("🟢 Good financial health!")
            elif prediction >= 15:
                st.warning("🟡 Moderate financial health. Room for improvement.")
            else:
                st.error("🔴 Low financial health score. Consider reviewing your expenses and savings.")

        with col_b:
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(["Score"], [prediction], color="#3b82f6", height=0.4)
            ax.barh(["Score"], [100 - prediction], left=[prediction], color="#e5e7eb", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Financial Health Score")
            ax.axvline(prediction, color="#1d4ed8", linestyle="--", lw=1.5)
            ax.set_title(f"Score: {prediction:.2f} / 100")
            ax.set_yticks([])
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("**Input Summary:**")
        summary = {
            "Monthly Income": f"${monthly_income:,.2f}",
            "Total Expenses": f"${total_expenses:,.2f}",
            "Savings Rate": f"{savings_rate*100:.1f}%",
            "Debt": f"${debt:,.2f}",
            "Expense/Income Ratio": f"{exp_ratio:.2f}",
        }
        st.table(pd.DataFrame(summary.items(), columns=["Metric", "Value"]).set_index("Metric"))
