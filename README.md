# CreditScore AI

A machine learning-based credit scoring application that predicts whether a customer is likely to have **Good Credit** or **Bad Credit** based on historical financial and personal information.

The project was developed as part of the **CodeAlpha Machine Learning Internship**.

---

## Live Demo

> Coming soon — the Streamlit deployment will be added here.

---

## Project Overview

Credit risk assessment is an important problem in the financial domain. This project uses supervised machine learning to classify customers according to their creditworthiness.

The application provides an interactive Streamlit interface where users can enter customer information and receive:

- Credit risk classification
- Good Credit probability
- Bad Credit probability
- Model performance information

The trained machine learning pipeline automatically handles numerical and categorical features before generating the prediction.

---

## Objectives

The main objectives of this project are:

- Build a credit classification model
- Perform data preprocessing and exploratory analysis
- Compare multiple machine learning algorithms
- Evaluate models using classification and ranking metrics
- Optimize the best-performing model
- Save the trained model for inference
- Build a professional interactive web application
- Deploy the application for public access

---

## Dataset

The project uses the **UCI Statlog (German Credit Data)** dataset.

Dataset characteristics:

- **Instances:** 1,000
- **Features:** 20
- **Target:** Creditworthiness
- **Missing values:** None
- **Good Credit:** 700
- **Bad Credit:** 300

The dataset contains financial, demographic, employment, credit history, savings, housing, and other customer-related attributes.

### Target Mapping

| Original Class | Model Label |
|---|---|
| 1 | Good Credit |
| 2 | Bad Credit |

For the machine learning pipeline, the target was converted to:

```text
1 → Good Credit
0 → Bad Credit