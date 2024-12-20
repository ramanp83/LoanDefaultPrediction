# LoanDefaultPrediction

# Loan Default Prediction Model

This project implements a machine learning model to predict loan defaults based on various financial features. It supports multiple model types, including Logistic Regression, Random Forest, and Gradient Boosting. The model processes the loan dataset, handles missing values, encodes categorical variables, and scales numerical features to train the model and make predictions.

## Features

- **Model Types**: Supports Logistic Regression, Random Forest, and Gradient Boosting models.
- **Preprocessing**: Handles missing values, scales numerical features, and encodes categorical features.
- **Evaluation**: Evaluates model performance using metrics such as Accuracy, Precision, Recall, F1-Score, and AUC-ROC.
- **Prediction**: Can make predictions on new data after training.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- openpyxl (for reading Excel files)

You can install the required dependencies using `pip`:

```bash
pip install pandas scikit-learn openpyxl
```
***File Structure***

LoanDefaultPrediction/
├── eda.ipynb
├── model_.py
├── model_selection.ipynb
├── train_data.xlsx
├── test_data.xlsx
