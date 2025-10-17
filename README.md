# Logistic Regression for Breast Cancer Classification

## Overview
This project demonstrates the application of Logistic Regression for classifying breast cancer diagnoses. Using a dataset of patient data, the model predicts whether a tumor is **benign** or **malignant** based on various features.

---

## Features
- **Data Loading and Preprocessing**:
  - Handles missing values and preprocesses the dataset for training.
  - Splits the data into training and testing sets.
- **Logistic Regression Model**:
  - Implements a Logistic Regression classifier for binary classification.
- **Model Evaluation**:
  - Evaluates the model using accuracy, confusion matrix, and classification report.
- **Visualization**:
  - Visualizes model performance metrics.

---

## Dataset
The dataset used is `breast_cancer.csv`, which contains:
- **Features**: Patient data such as cell size, shape, and other characteristics.
- **Target**: Binary labels indicating whether the tumor is benign (0) or malignant (1).

Place the dataset in the same directory as the notebook or update the file path in the code.

---

## Getting Started

### Prerequisites
To run this project, ensure you have:
- Python 3.8+
- Jupyter Notebook
- Required Python libraries (listed below).

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-logistic-regression.git
   cd breast-cancer-logistic-regression
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook logistic_regression_breast_cancer.ipynb
   ```
2. Run the cells sequentially to:
   - Load and preprocess the dataset.
   - Train the Logistic Regression model.
   - Evaluate its performance.

---

## Libraries Used
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For model training and evaluation.
- **Matplotlib**: For visualizing results.

Install these libraries using:
```bash
pip install pandas numpy scikit-learn matplotlib
```





