# Credit Card Fraud Detection – Predictive Modeling Project


## Project Overview

This project focuses on building a **predictive machine learning model** to detect fraudulent credit card transactions using anonymized transaction data. The challenge is especially interesting due to the **extremely imbalanced dataset**, where fraudulent transactions are a very small fraction of the total transactions.

This project walks through the complete ML lifecycle — from data cleaning, EDA, and preprocessing to model building, evaluation, and final prediction using different supervised learning techniques.

---

## 📂 Dataset

The dataset used in this project is from **Kaggle’s Credit Card Fraud Detection** challenge. It contains transactions made by European cardholders in September 2013.

- Total observations: **284,807**
- Fraudulent transactions: **492**
- Features: 30 (Anonymized PCA features + `Time`, `Amount`, and `Class`)

> 📥 Dataset Source: [Credit Card Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📈 Project Workflow

1. **Importing Dependencies**
   - Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
2. **Exploratory Data Analysis (EDA)**
   - Distribution of classes (fraud vs normal)
   - Correlation heatmaps and feature importance
   - Insights from `Time` and `Amount` features
3. **Data Preprocessing**
   - Feature scaling using `StandardScaler` and `RobustScaler`
   - Handling class imbalance via **undersampling** and **SMOTE (oversampling)**
4. **Model Building**
   - Logistic Regression
   - Random Forest
   - XGBoost
   - Support Vector Machine
   - Decision Tree Classifier
5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix visualization
   - Cross-validation for robustness
6. **Final Results**
   - Model comparison
   - Best-performing algorithm identified
   - Discussion on trade-offs between recall and precision

---

## 🚀 Key Findings

- The dataset is **highly imbalanced** (~0.17% fraud cases), requiring custom sampling techniques.
- **Precision and Recall** were prioritized over accuracy due to the cost of false negatives.
- Among various models, **Random Forest** and **XGBoost** gave the most balanced and robust performance.
- SMOTE-based oversampling improved minority class detection but needed tuning to avoid overfitting.

---

## 🛠️ Technologies Used

| Component | Technology |
|----------|------------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Seaborn, Matplotlib |
| ML Models | Scikit-learn, XGBoost |
| Model Evaluation | Sklearn metrics, Confusion Matrix, AUC-ROC |
| IDE | Jupyter Notebook |
| Source Control | Git & GitHub |

---

## 💡 How to Run

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/creditcard-fraud-detection.git
    cd creditcard-fraud-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` inside the project folder.

4. Run the notebook:
    ```bash
    jupyter notebook creditcard-fraud-detection-predictive-models.ipynb
    ```

---

## ✅ Conclusion

This project demonstrated how machine learning can be applied effectively to imbalanced classification problems such as credit card fraud detection. Techniques like **SMOTE**, **undersampling**, and careful model evaluation are essential for such domains. The project provides a solid foundation to build real-world financial fraud detection systems.

---


Feel free to connect or contribute via GitHub!

