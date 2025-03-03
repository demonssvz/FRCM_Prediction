import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Streamlit Page Config
st.set_page_config(page_title="Composite Strength Predictor of Tensile Strength", layout="wide")

# Title
st.title("🔬 Composite Material Strength Prediction (T1)")

# Sidebar - File Upload
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

@st.cache_data
def load_and_preprocess_data(file):
    if file is None:
        return None, None
    
    xls = pd.ExcelFile(file)
    df = pd.read_excel(xls, sheet_name='Sheet1')

    df_cleaned = df.iloc[5:].reset_index(drop=True)
    df_cleaned.columns = df_cleaned.iloc[0]
    df_cleaned = df_cleaned[1:].reset_index(drop=True)
    df_cleaned = df_cleaned.iloc[:, 2:]

    df_cleaned.columns = [
        "Fiber_type", "Fiber_volume_ratio", "Tensile_strength_fiber_yarn",
        "Elastic_modulus_yarn", "Tensile_strength_mortar", "Elastic_modulus", "T1", "Tu"
    ]

    numeric_columns = [
        "Fiber_volume_ratio", "Tensile_strength_fiber_yarn", "Elastic_modulus_yarn",
        "Tensile_strength_mortar", "Elastic_modulus", "T1"
    ]
    
    for col in numeric_columns:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    encoder = LabelEncoder()
    df_cleaned["Fiber_type"] = encoder.fit_transform(df_cleaned["Fiber_type"].astype(str))

    return df_cleaned, encoder

@st.cache_data
def train_and_plot(df_cleaned, model_name):
    X = df_cleaned.drop(columns=["T1", "Tu"])  
    y_t1 = df_cleaned["T1"]

    # Split into training and testing
    X_train, X_test, y_t1_train, y_t1_test = train_test_split(
        X, y_t1, test_size=0.2, random_state=42
    )

    # Use StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter-Tuned Models
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    }

    model_t1 = models[model_name]
    model_t1.fit(X_train_scaled, y_t1_train)

    # Save model & test data
    joblib.dump(model_t1, "model_t1.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(encoder, "encoder.pkl")
    joblib.dump(y_t1_test, "y_t1_test.pkl")  
    joblib.dump(X_test_scaled, "X_test_scaled.pkl")  
    joblib.dump(X_test, "X_test.pkl")  # ✅ Save X_test for visualization

    return model_t1, scaler

# Load Data
if uploaded_file:
    df_cleaned, encoder = load_and_preprocess_data(uploaded_file)

    if df_cleaned is not None:
        st.subheader("📊 Dataset Preview")
        st.dataframe(df_cleaned.head())

        model_option = st.selectbox("Choose a Model", ["Random Forest", "Gradient Boosting", "XGBoost"])

        if st.button("🚀 Train & Save Model"):
            with st.spinner("Training in progress..."):
                train_and_plot(df_cleaned, model_option)
                st.success("✅ Model trained and saved successfully!")

# Prediction Section (T1 Only)
st.subheader("🔮 Predict Strength (T1) of New Material")

try:
    if all(os.path.exists(f) for f in ["model_t1.pkl", "scaler.pkl", "encoder.pkl", "y_t1_test.pkl", "X_test_scaled.pkl", "X_test.pkl"]):
        model_t1 = joblib.load("model_t1.pkl")
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("encoder.pkl")

        col1, col2 = st.columns(2)

        with col1:
            fiber_type = st.selectbox("Fiber Type", ["CARBON", "GLASS", "KEVLAR"])
            fiber_volume_ratio = float(st.text_input("Fiber Volume Ratio (Decimal format)", "0.00462"))
            tensile_strength_fiber_yarn = st.number_input("Tensile Strength of Fiber Yarn", min_value=0, step=1, value=2125)

        with col2:
            elastic_modulus_yarn = st.number_input("Elastic Modulus of Yarn", min_value=0, step=100, value=200000)
            tensile_strength_mortar = st.number_input("Tensile Strength of Mortar", min_value=0.0, step=0.1, value=5.2)
            elastic_modulus = st.number_input("Elastic Modulus of Matrix", min_value=0, step=100, value=12000)

        fiber_type_encoded = encoder.transform([fiber_type])[0]
        input_data = np.array([[fiber_type_encoded, fiber_volume_ratio, tensile_strength_fiber_yarn, 
                                elastic_modulus_yarn, tensile_strength_mortar, elastic_modulus]])

        input_data_scaled = scaler.transform(input_data)

        t1_pred = model_t1.predict(input_data_scaled)[0]

        st.success(f"🎯 **Predicted T1:** {t1_pred:.4f}")

        if st.button("📊 Show Results & Discussion"):
            y_t1_test = joblib.load("y_t1_test.pkl")
            X_test_scaled = joblib.load("X_test_scaled.pkl")
            X_test = joblib.load("X_test.pkl")  
            y_t1_pred = model_t1.predict(X_test_scaled)

            # Metrics
            mae = mean_absolute_error(y_t1_test, y_t1_pred)
            mse = mean_squared_error(y_t1_test, y_t1_pred)
            r2 = r2_score(y_t1_test, y_t1_pred)

            st.write(f"📏 **MAE:** {mae:.4f}")
            st.write(f"📏 **MSE:** {mse:.4f}")
            st.write(f"📏 **R² Score:** {r2:.4f}")

            # Separate Graphs
            st.subheader("📊 Graphs & Insights")

            # Actual vs Predicted
            fig, ax = plt.subplots()
            ax.scatter(y_t1_test, y_t1_pred, alpha=0.6)
            ax.plot([min(y_t1_test), max(y_t1_test)], [min(y_t1_test), max(y_t1_test)], color='red', linestyle='dashed')
            ax.set_title("T1: Actual vs Predicted")
            st.pyplot(fig)

            # Residual Distribution
            residuals = y_t1_test - y_t1_pred
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Residual Distribution")
            st.pyplot(fig)

            # Feature Importance
            feature_importance = model_t1.feature_importances_
            fig, ax = plt.subplots()
            sns.barplot(x=feature_importance, y=X_test.columns, ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

    else:
        st.warning("⚠️ Please train and save the model first!")

except Exception as e:
    st.error(f"❌ Error: {e}")
