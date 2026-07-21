# Polycystic Ovary Syndrome (PCOS) Risk Prediction Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)

An end-to-end Biomedical Machine Learning project that integrates clinical data preprocessing, model training, and a web-based clinical decision support tool for **Polycystic Ovary Syndrome (PCOS)** risk assessment.

---

## Project Overview

Polycystic Ovary Syndrome (PCOS) is one of the most common endocrine and reproductive disorders in women. Early identification of clinical risk factors is essential for preventing long-term metabolic and fertility complications.

Developed as part of my **B.Sc. in Bioengineering** portfolio, this project bridges computational data analysis with clinical biomarkers. It processes merged patient datasets, cleans conflictive hormonal parameters, trains a **Random Forest Classifier**, and provides a real-time risk assessment interface powered by **Streamlit**.

---

## Bioengineering & ML Highlights

* **Biomarker Selection:** Selected key clinical indicators validated in medical literature (Age, BMI, Anti-Müllerian Hormone (AMH), Follicle Count in both ovaries, Menstrual Cycle Regularity, Weight Gain, and Hirsutism).
* **Data Cleaning & Imputation:** Handled missing values via median imputation and performed dynamic column cleaning to resolve string coercion issues in hormonal variables (`AMH` and `II beta-HCG`).
* **Stratified Classification:** Utilized an 80/20 train-test split with stratification to maintain diagnosis class distribution and prevent model bias.
* **Model Export & Inference:** Serialized the trained model using `joblib` for low-latency predictions in the web frontend.

---

## Repository Structure

```text
pcos-ml-predictor/
├── data/
│   ├── PCOS_data_without_infertility.xlsx
│   └── PCOS_infertility.csv
├── models/
│   └── rf_pcos_model.joblib
├── .gitignore
├── PCOS_app.py
├── pcos.py
├── README.md
└── requirements_PCOS.txt
```
---

## Execution Guide

### 1. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/aitor-calero-bioeng/PCOS-pipeline
cd PCOS-pipeline
pip install -r requirements.txt
```
### 2. Model Training
To run the data cleaning pipeline and train the Random Forest classifier from scratch:
python PCOS.py

### 3. Run Web Dashboard
To launch the interactive Streamlit clinical application locally:
streamlit run PCOS_app.py







