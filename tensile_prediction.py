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

 
st.set_page_config(page_title="Composite Strength Predictor", layout="wide")
 
st.title("🔬 Composite Material Strength Prediction")

 
tab1, tab2, tab3 = st.tabs(["📂 Upload Data", "🚀 Train Model & Results", "🔮 Predict"])
 
with tab1:
    st.header("📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (.xlsx)", type=["xlsx"])

    if uploaded_file:
        st.success("✅ File uploaded successfully!")

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

    if uploaded_file:
        df_cleaned, encoder = load_and_preprocess_data(uploaded_file)

        if df_cleaned is not None:
            st.subheader("📊 Data Preview")
            st.dataframe(df_cleaned.head())

            with st.expander("📈 Dataset Statistics"):
                st.write(df_cleaned.describe())

            with st.expander("🔍 Missing Values"):
                st.write(df_cleaned.isnull().sum())

# ---- Train Model & Results ----
with tab2:
    st.header("🚀 Train Model & View Results")

    if uploaded_file:
        model_option = st.selectbox("Choose a Model", ["Random Forest", "Gradient Boosting", "XGBoost"])
        train_button = st.button("Train & Save Model")

        @st.cache_data
        def train_and_save_model(df_cleaned, model_name):
            X = df_cleaned.drop(columns=["T1", "Tu"])  
            y_t1 = df_cleaned["T1"]

            X_train, X_test, y_train, y_test = train_test_split(X, y_t1, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42),
                "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
            }

            model = models[model_name]
            model.fit(X_train_scaled, y_train)

            joblib.dump(model, "model_t1.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(encoder, "encoder.pkl")
            joblib.dump(y_test, "y_t1_test.pkl")
            joblib.dump(X_test_scaled, "X_test_scaled.pkl")
            joblib.dump(X_test, "X_test.pkl") 

            return model, y_test, X_test_scaled, X_test

        if train_button:
            with st.spinner("Training in progress..."):
                model, y_test, X_test_scaled, X_test = train_and_save_model(df_cleaned, model_option)
                st.success("✅ Model trained and saved!")

                # ---- Display Model Performance ----
                y_pred = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.subheader("📊 Model Performance")
                st.write(f"📏 **MAE:** {mae:.4f}")
                st.write(f"📏 **MSE:** {mse:.4f}")
                st.write(f"📏 **R² Score:** {r2:.4f}")

                # ---- Graphs ----
                st.subheader("📊 Graphs & Insights")

                # Actual vs Predicted Plot
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
                ax.set_title("T1: Actual vs Predicted")
                st.pyplot(fig)

                # Residual Distribution
                residuals = y_test - y_pred
                fig, ax = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax)
                ax.set_title("Residual Distribution")
                st.pyplot(fig)

                # Feature Importance
                feature_importance = model.feature_importances_
                fig, ax = plt.subplots()
                sns.barplot(x=feature_importance, y=X_test.columns, ax=ax)
                ax.set_title("Feature Importance")
                st.pyplot(fig)

# ---- Prediction Section ----
with tab3:
    st.header("🔮 Predict T1 Strength")

    if os.path.exists("model_t1.pkl"):
        model = joblib.load("model_t1.pkl")
        scaler = joblib.load("scaler.pkl")
        encoder = joblib.load("encoder.pkl")

        fiber_type = st.selectbox("Fiber Type", ["CARBON", "GLASS", "BASALT"])
        fiber_volume_ratio = float(st.text_input("Fiber Volume Ratio (Decimal format)", "0.00462"))
        tensile_strength_fiber_yarn = st.number_input("Tensile Strength of Fiber Yarn", min_value=0, step=1, value=2125)
        elastic_modulus_yarn = st.number_input("Elastic Modulus of Yarn", min_value=0, step=100, value=200000)
        tensile_strength_mortar = st.number_input("Tensile Strength of Mortar", min_value=0.0, step=0.1, value=5.2)
        elastic_modulus = st.number_input("Elastic Modulus of Matrix", min_value=0, step=100, value=12000)

        fiber_type_encoded = encoder.transform([fiber_type])[0]
        input_data = np.array([[fiber_type_encoded, fiber_volume_ratio, tensile_strength_fiber_yarn, 
                                elastic_modulus_yarn, tensile_strength_mortar, elastic_modulus]])

        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]

        st.success(f"🎯 **Predicted T1:** {prediction:.4f}")
