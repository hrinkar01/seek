# Seek 🚀

Seek is an automated machine learning assistant built with Scikit-Learn that helps users quickly train and evaluate machine learning models on tabular datasets with minimal manual preprocessing.

## Features

### 📂 Automatic Dataset Loading

* Loads any CSV dataset.
* Allows users to choose the target column to predict.

### 🧹 Automatic Data Cleaning

* Removes rows with missing target values.
* Detects and drops useless columns such as:

  * ID columns
  * High-cardinality metadata columns (e.g., names)
  * Near-constant columns

### 🔍 Feature Detection

* Automatically identifies:

  * Numerical features
  * Categorical features

### ⚙️ Missing Value Handling

* Numerical features:

  * Filled using the median value.
* Categorical features:

  * Filled with `"unknown"`.

### 🔤 Automatic Encoding

* Applies One-Hot Encoding to categorical features.

### 📏 Automatic Scaling

* Standardizes numerical features using StandardScaler.

### 🏗️ Scikit-Learn Pipelines

* Uses Pipeline and ColumnTransformer to ensure preprocessing and model training happen in a clean, reusable workflow.

### 🤖 Automatic Problem Detection

* Detects whether the task is:

  * Classification
  * Regression

### 🧠 Multiple Model Training

#### Classification

* Logistic Regression
* Random Forest Classifier
* HistGradientBoosting Classifier

#### Regression

* Linear Regression
* Random Forest Regressor
* HistGradientBoosting Regressor

### 📊 Cross Validation

* Evaluates every model using 5-Fold Cross Validation.
* Reports average performance scores.

### 🏆 Model Comparison

* Compares multiple models and displays their evaluation scores to help identify the best-performing algorithm.

## Workflow

Dataset
→ Data Cleaning
→ Feature Detection
→ Missing Value Handling
→ Encoding
→ Scaling
→ Preprocessing Pipeline
→ Model Training
→ Cross Validation
→ Model Comparison

## Goal

Seek aims to simplify the machine learning workflow by automating common preprocessing and model evaluation tasks, allowing users to focus on understanding their data and selecting the best model.
