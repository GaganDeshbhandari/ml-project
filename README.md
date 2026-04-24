# Bias-Aware-News-Prioritizer — ML Pipeline

Implementation of the research paper **"Analyzing Biases in Perception of Truth in News Stories"** (Babaei et al., 2021).

---

## Overview

This repository contains the ML pipeline that trains and saves models to prioritize news claims for fact-checking based on how users perceive them.

---

## Repository Structure
```
mini-project/
├── scripts.ipynb              
├── datasets/
│   ├── claims_dataset.csv
│   └── user_perception_dataset.csv
├── random_forest_model.pkl    
├── random_forest_model_o1.pkl 
└── scaler.pkl                 
```
---

## Datasets

| File | Description |
|---|---|
| `claims_dataset.csv` | 171 claims with ground truth labels |
| `user_perception_dataset.csv` | 20 user ratings per claim |

---

## Key Concepts

| Term | Meaning |
|---|---|
| GTL | Numeric value of fact-checked label per claim |
| PTL | Average of all user ratings per claim |
| TPB | How wrong the crowd is about a claim |
| FPB | Users thought claim was more true than it is |
| FNB | Users thought claim was more false than it is |

---

## Three Objectives

| Objective | Goal | Approach |
|---|---|---|
| O1 | Remove false claims | Classify as True or False |
| O2 | Correct misperceptions | Classify as High or Low TPB |
| O3 | Reduce disagreement | Rank by variance of user ratings |

---

## Model Results

| Model | O1 Accuracy | O2 Accuracy |
|---|---|---|
| Linear SVM | 99% | 51% |
| Naive Bayes | 98% | 70% |
| Logistic Regression | 99% | 53% |
| **Random Forest** | **98%** | **91%** |

> ✅ Random Forest selected as final model for both O1 and O2.

---

## Saved Model Files

| File | Purpose |
|---|---|
| `random_forest_model_o1.pkl` | Predicts True or False for new claims |
| `random_forest_model.pkl` | Predicts High or Low TPB for new claims |
| `scaler.pkl` | Must be applied to new data before prediction |

---

## How to Run

1. Open `scripts.ipynb` in Google Colab
2. Mount Google Drive
3. Run all cells in order

---

## Tech Stack

- Python, Pandas, Scikit-learn, Scipy, Matplotlib, Joblib, Google Colab
