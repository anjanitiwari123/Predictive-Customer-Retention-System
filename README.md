# 📊 Customer Churn Prediction & Retention Intelligence

An end-to-end machine learning system that predicts the probability of customer churn and identifies high-risk customers for proactive retention strategies.

The project covers the complete ML lifecycle — from data analysis and preprocessing to model evaluation, decision-threshold optimization, model serialization, and deployment through an interactive Streamlit application.

---

## 🚀 Project Overview

Customer churn is a major business problem for subscription-based businesses. Identifying customers who are likely to leave allows companies to intervene before churn occurs.

This project builds a supervised machine learning pipeline that:

- Analyzes customer demographics, services, contracts, and billing behavior
- Preprocesses numerical and categorical features
- Trains and evaluates classification models
- Selects the final model based on validation performance
- Optimizes the classification decision threshold instead of blindly using `0.50`
- Generates churn and retention probabilities
- Provides an interactive Streamlit interface
- Converts model predictions into actionable retention recommendations

---

## 🎯 Business Problem

The objective is to answer:

> **"Which customers are likely to churn, and how confidently does the model predict their churn risk?"**

A binary classification model predicts:

- `0` → Customer is likely to stay
- `1` → Customer is likely to churn

Instead of relying only on a binary prediction, the application exposes the predicted churn probability and compares it against a business-oriented decision threshold.

This allows the system to distinguish between:

- **Low Churn Risk**
- **High Churn Risk**

---

## 🧠 Key ML Features

The model uses customer-level behavioral and service information including:

### Customer Profile
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure

### Service Information
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies

### Contract & Billing
- Contract Type
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

Customer ID is intentionally excluded because it does not provide meaningful predictive information for churn modeling.

---

## 🔄 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Quality Analysis
        ↓
Exploratory Data Analysis
        ↓
Feature Analysis
        ↓
Data Preprocessing
        ↓
Train / Validation / Test Strategy
        ↓
Classification Models
        ↓
Model Evaluation
        ↓
Decision Threshold Optimization
        ↓
Final Model Selection
        ↓
Pipeline Serialization
        ↓
Streamlit Deployment
```

---

## 🔍 Exploratory Data Analysis

The analysis investigates:

- Dataset structure and data types
- Missing values
- Duplicate records
- Target distribution
- Numerical feature distributions
- Categorical feature distributions
- Churn patterns across customer segments
- Relationship between tenure and churn
- Contract type vs churn
- Billing behavior vs churn
- Service usage vs churn

The EDA is used not only for visualization but also to understand which customer characteristics are associated with churn and to guide preprocessing/modeling decisions.

---

## ⚙️ Data Preprocessing

The project uses a machine learning preprocessing pipeline so that the same transformations are applied consistently during training and inference.

The saved pipeline handles preprocessing internally, allowing the Streamlit application to pass raw customer fields directly to the trained model. fileciteturn2file1L153-L177

This reduces the risk of training-serving inconsistencies.

---

## 🤖 Model Development

Multiple classification approaches were considered during the modeling stage.

The final deployed model is:

**Logistic Regression**

The serialized model pipeline is loaded by the Streamlit application rather than retraining the model during every prediction. fileciteturn2file1L30-L38

### Why Logistic Regression?

Logistic Regression provides:

- Strong baseline performance for binary classification
- Fast inference
- Interpretable coefficients
- Probability-based predictions
- A natural fit for threshold optimization
- Lightweight deployment

---

## 📈 Model Evaluation

The final deployed model achieves:

| Metric | Result |
|---|---:|
| Model | Logistic Regression |
| Test ROC-AUC | **0.839** |
| Decision Threshold | **0.55** |

The application displays the test ROC-AUC and selected decision threshold directly from the saved model metadata. fileciteturn2file1L37-L43

### ROC-AUC

ROC-AUC measures how effectively the model ranks customers who churn above customers who stay across different classification thresholds.

A test ROC-AUC of **0.839** indicates strong discriminatory performance on the held-out test data.

---

## 🎚️ Decision Threshold Optimization

A key part of the project is that the final prediction does **not** simply use the default `0.50` threshold.

The deployed threshold is:

```text
0.55
```

The application classifies a customer as high-risk when:

```text
Churn Probability >= 0.55
```

The threshold was selected using **out-of-fold F1 optimization on the training data**, rather than optimizing directly on the held-out test set. This keeps the test set isolated for final evaluation. fileciteturn2file1L180-L183

This is particularly useful for churn prediction because the business may prefer a different balance between identifying potential churners and avoiding unnecessary retention interventions.

---

## 🧮 Prediction Logic

For every customer, the system generates:

```text
Churn Probability
Stay Probability
```

The probabilities are complementary:

```text
Stay Probability = 1 - Churn Probability
```

The application then compares the churn probability with the selected threshold.

### Example

```text
Churn Probability = 69.4%
Decision Threshold = 55%

69.4% >= 55%

→ High Churn Risk
```

Whereas:

```text
Churn Probability = 51.0%
Decision Threshold = 55%

51.0% < 55%

→ Low Churn Risk
```

---

## 💡 Business Recommendations

When a customer is classified as high risk, the application recommends proactive retention actions such as:

- Personalized offers
- Customer support assistance
- Contract incentives
- Targeted retention campaigns

These recommendations are intended as business actions following model prediction, rather than being generated by a separate generative AI system.

---

## 🖥️ Streamlit Application

The project includes an interactive Streamlit application where users can enter customer information and obtain an immediate churn prediction.

The application:

1. Accepts customer profile and service information
2. Builds the raw feature row
3. Loads the pre-trained model pipeline
4. Generates churn probability
5. Calculates stay probability
6. Applies the optimized threshold
7. Displays the customer risk level
8. Provides retention-oriented recommendations for high-risk customers

The application loads the serialized pipeline and metadata at runtime; **model training does not happen inside the deployed application**. fileciteturn2file1L30-L38

---

## 🏗️ Project Structure

```text
Customer-Churn-Prediction/
│
├── Customer_Churn_Analysis.ipynb
│
├── app.py
│
├── models/
│   ├── best_churn_model.pkl
│   └── model_metadata.json
│
├── requirements.txt
│
└── README.md
```

> The exact repository structure may vary depending on the files included in the GitHub repository.

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Analysis
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Machine Learning
- Scikit-learn

### Model Deployment
- Streamlit

### Model Serialization
- Joblib
- JSON

### Development Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/anjanitiwari123/Loan_Approval_System.git
cd Loan_Approval_System
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Streamlit application

```bash
streamlit run app.py
```

The application will open in the browser.

---

## 📊 Application Output

The application displays:

```text
Customer Churn Predictor

Model: Logistic Regression
Test ROC-AUC: 0.839
Decision threshold: 0.55

Churn Probability: XX.X%
Stay Probability: XX.X%

Risk Classification:
High Churn Risk / Low Churn Risk
```

The application also provides retention recommendations for customers classified as high risk. fileciteturn2file1L180-L213

---

## 🔐 Model Serving Design

The deployed application follows a simple inference architecture:

```text
User Input
    ↓
Pandas DataFrame
    ↓
Saved ML Pipeline
    ↓
Probability Prediction
    ↓
Optimized Threshold
    ↓
Risk Classification
    ↓
Business Recommendation
```

The model is loaded using Streamlit's resource caching mechanism, avoiding unnecessary model reloads during application reruns. fileciteturn2file1L30-L38

---

## 📌 Important ML Engineering Practices

This project demonstrates several practical machine learning concepts beyond simply training a classifier:

- End-to-end preprocessing pipeline
- Separation of training and inference
- Held-out test evaluation
- ROC-AUC based model assessment
- Probability-based predictions
- Out-of-fold threshold optimization
- Business-oriented classification threshold
- Serialized model artifacts
- Reproducible inference pipeline
- Interactive ML deployment

---

## ⚠️ Limitations

The model should be treated as a decision-support system rather than a guarantee of future customer behavior.

Potential limitations include:

- Customer behavior can change over time
- Historical patterns may not represent future behavior
- Model performance depends on the quality and representativeness of the training data
- Probability estimates should not automatically be interpreted as guaranteed churn likelihood
- Retention recommendations require business validation and experimentation

---

## 🔮 Future Improvements

Potential extensions include:

- Probability calibration
- Cost-sensitive threshold optimization
- Model monitoring
- Prediction drift monitoring
- SHAP-based individual customer explanations
- Automated retention campaign prioritization
- Cost/benefit-based retention optimization
- Batch prediction for large customer datasets
- A/B testing of retention strategies
- Time-based validation for more realistic production evaluation

---

## 🎓 Skills Demonstrated

This project demonstrates practical experience in:

**Python • Pandas • NumPy • Scikit-learn • Exploratory Data Analysis • Feature Engineering • Classification • Logistic Regression • ROC-AUC • F1 Optimization • Cross-Validation • Decision Threshold Optimization • Model Serialization • ML Pipelines • Streamlit • Git • GitHub**

---

## 👨‍💻 Project Highlights

### Machine Learning
- Built an end-to-end customer churn classification system
- Evaluated classification performance using ROC-AUC
- Achieved **0.839 Test ROC-AUC**
- Implemented probability-based churn prediction
- Optimized the decision threshold using out-of-fold training predictions

### ML Engineering
- Serialized the complete model pipeline
- Separated model training from inference
- Reused the same preprocessing pipeline during deployment
- Stored model configuration and evaluation metadata separately

### Deployment
- Developed an interactive Streamlit prediction interface
- Provides real-time churn probability and risk classification
- Converts predictions into actionable retention recommendations

---

## ⭐ Key Takeaway

This project goes beyond a basic churn prediction notebook by combining:

**Data Analysis → Model Development → Robust Evaluation → Threshold Optimization → Model Serialization → Interactive Deployment → Business Action**

The goal is not only to predict **who may churn**, but to provide a practical framework for identifying high-risk customers early enough for targeted retention efforts.
