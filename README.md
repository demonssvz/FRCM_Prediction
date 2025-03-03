# 🔬 Composite Strength Predictor (T1)  
**A Machine Learning-powered Streamlit app to predict the tensile strength (T1) of composite materials based on fiber properties.**  

![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![Python](https://img.shields.io/badge/Python-3.8-blue) ![Machine%20Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-orange)

### 🔗 **Live App:**  
✅ **You Can Access the Project Here:**  
👉 [https://frcm-prediction.streamlit.app/](https://frcm-prediction.streamlit.app/)  

---

## 🚀 **About This Project**
Composite materials are widely used in aerospace, automotive, and construction industries. This **web-based machine learning app** predicts the **T1 strength** of composite materials based on **fiber type, fiber volume ratio, and mechanical properties**.

Using **Random Forest, Gradient Boosting, and XGBoost**, this app provides a user-friendly interface for:
- **Uploading datasets**
- **Training ML models**
- **Evaluating results**
- **Making predictions for new composite materials**

---

## 📌 **Features**
✅ **Upload Your Dataset** (`.xlsx` format)  
✅ **Data Preprocessing & Cleaning** (Handles missing values, encoding, etc.)  
✅ **Train ML Models** (Random Forest, Gradient Boosting, XGBoost)  
✅ **Performance Metrics** (MAE, MSE, R² Score)  
✅ **Graphical Insights** (Actual vs Predicted, Residuals, Feature Importance)  
✅ **Make Predictions** (Enter fiber properties to predict T1 strength)  
✅ **Deployable on Streamlit Cloud**  

---

## 🛠 **Technologies Used**
- **Python 3.8+**
- **Streamlit** (Web UI)
- **Pandas, NumPy** (Data Handling)
- **Scikit-Learn** (ML Models)
- **XGBoost** (Boosting Algorithm)
- **Matplotlib, Seaborn** (Visualization)
- **Joblib** (Model Saving)

---

## 🔧 **Installation & Setup**
### **1️⃣ Clone This Repository**
```bash
git clone https://github.com/demonssvz/FRCM_Prediction.git
cd composite-strength-predictor
```
2️⃣ Install Dependencies
Ensure you have Python 3.8+ installed. Then, install required packages:

```bash

pip install -r requirements.txt
```
3️⃣ Run the Streamlit App
```bash
 
streamlit run app.py
```
This will start a local server, and you can access the app in your browser.



## 🤝 Contributing
Want to improve this project? Contributions are welcome!
To contribute:

Fork this repository.

Create a new branch (feature-branch).

Commit your changes (git commit -m "Added a new feature").

Push to GitHub and create a Pull Request.
