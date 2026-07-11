# AI-ML-and-GEN-AI-Track-Project
# Credit Card Approval Prediction

Domain: Finance

Team Lead: Ankitha Vana

Mentor: Not assigned yet

Prepared on: July 1, 2026

## Project Overview

Banks and financial institutions receive thousands of credit card applications every day. This project automates the initial screening process using machine learning models trained on applicant and credit history data. The selected model is integrated with a Flask web application and can be deployed through IBM Watson Machine Learning.

## What Is Included

- Template-matched phase deliverables as PDFs
- Machine learning training pipeline
- Logistic Regression, Decision Tree, Random Forest, and XGBoost comparison
- Saved best model in `models/credit_card_model.joblib`
- Flask web app for approval/rejection prediction
- HTML/CSS user interface
- IBM Watson Machine Learning deployment notes

## Repository Structure

```text
Credit_Card_Approval_Prediction_Project/
  1. Brainstorming & Ideation/
  2. Requirement Analysis/
  3. Project Design Phase/
  4. Project Planning Phase/
  5. Project Development Phase/
  6.Project Testing/
  7.Project Documentation/
  8.Project Demonstration/
  app.py
  train_model.py
  src/
  templates/
  static/
  data/
  models/
  reports/
  requirements.txt
  watson_deployment_notes.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python train_model.py
```

If `data/application_record.csv` and `data/credit_record.csv` are present, the project trains using those files. If they are not present, the project trains using a generated demo dataset so the app works immediately.

Training outputs:

- `models/credit_card_model.joblib`
- `reports/model_metrics.csv`
- `reports/training_summary.json`

## Run the Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Demo and GitHub Links

- Demo link: https://drive.google.com/file/d/1oGAxv3xLB760QIDgQyioP692i0qjPPQ6/view?usp=sharing
- GitHub link: https://github.com/AnkithaVana/AI-ML-and-GEN-AI-Track-Project
