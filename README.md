# CreditScore AI

### Machine Learning Based Credit Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Ashishcoder01/CodeAlpha_Credit_Scoring_Model)

CreditScore AI is a machine learning application that predicts customer creditworthiness using historical financial and personal information.

The project compares multiple classification algorithms, performs cross-validation and hyperparameter tuning, and provides an interactive Streamlit interface for credit risk prediction.

> Developed as part of the **CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model**.

---

## Live Demo

**Try CreditScore AI:**

https://creditscore-ai.streamlit.app/

**Source Code:**

https://github.com/Ashishcoder01/CodeAlpha_Credit_Scoring_Model

---

## Overview

Credit risk assessment is an important machine learning problem in the financial domain.

This project uses supervised classification techniques to predict whether a customer is likely to have:

- Good Credit
- Bad Credit

The application accepts customer information through a user-friendly interface, applies the same preprocessing pipeline used during model training, and generates a credit risk prediction with probability estimates.

---

## Key Features

- Machine learning based credit risk prediction
- Interactive Streamlit web application
- Numerical feature standardization
- Categorical feature encoding
- Multiple classification models
- Stratified train/test split
- 5-fold cross-validation
- Random Forest hyperparameter tuning
- ROC-AUC based model comparison
- Credit probability estimation
- Human-readable input fields
- Saved production-ready model pipeline
- Model metadata storage
- GitHub-based project structure
- Cloud deployment using Streamlit Community Cloud

---

## Dataset

The project uses the **UCI Statlog (German Credit Data)** dataset.

### Dataset Information

| Property | Value |
|---|---:|
| Instances | 1,000 |
| Features | 20 |
| Target Classes | 2 |
| Good Credit | 700 |
| Bad Credit | 300 |
| Missing Values | None |

The dataset contains financial and personal attributes related to customers, including credit history, credit amount, savings, employment, housing, age, existing credits, and other financial information.

### Target Mapping

| Original Class | Model Label |
|---|---|
| `1` | Good Credit |
| `2` | Bad Credit |

For model training:

```text
1 → Good Credit
0 → Bad Credit