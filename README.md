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

---

## Results
- The Logistic Regression model achieved high accuracy in predicting breast cancer diagnoses.
- Detailed evaluation metrics are available in the notebook.



import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1️⃣ Connect to your MySQL database ---
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",   # 🔒 change this
    database="your_db",         # 🔒 change this
    connection_timeout=600
)

# --- 2️⃣ Run your SQL query ---
query = """
SELECT step_name, duration
FROM your_table
WHERE duration IS NOT NULL
"""
df = pd.read_sql(query, conn)
conn.close()

print("✅ Loaded", len(df), "rows")

# --- 3️⃣ Convert duration to hours (was in minutes) ---
df["duration"] = df["duration"] / 60  # ⚠️ If in seconds, divide by 3600 instead

# --- 4️⃣ Compute count (nombre) and average (moyenne) per step ---
stats = df.groupby("step_name")["duration"].agg(["count", "mean"]).reset_index()
stats.rename(columns={"count": "nombre", "mean": "moyenne"}, inplace=True)

print("📊 Moyenne (heures) par étape :")
print(stats)

# --- 5️⃣ Create the figure ---
plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

# --- 5a️⃣ Left chart — Average duration curve ---
plt.subplot(1, 2, 1)
plt.plot(stats["step_name"], stats["moyenne"], marker='o', color='blue', linewidth=2)
for i, row in stats.iterrows():
    plt.text(row["step_name"], row["moyenne"], f"{row['moyenne']:.2f} h", ha="center", va="bottom", fontsize=9)
plt.title("Durée moyenne par étape (en heures)")
plt.xlabel("Nom de l’étape")
plt.ylabel("Durée moyenne (heures)")
plt.grid(True)

# --- 5b️⃣ Right chart — Distribution of all durations ---
plt.subplot(1, 2, 2)
sns.kdeplot(
    data=df,
    x="duration",
    hue="step_name",
    fill=True,
    alpha=0.3,
    linewidth=2
)
sns.stripplot(
    data=df,
    x="duration",
    y="step_name",
    alpha=0.4,
    color="black",
    jitter=True
)
plt.title("Distribution des durées par étape (en heures)")
plt.xlabel("Durée (heures)")
plt.ylabel("Étape")

plt.tight_layout()

# --- 6️⃣ Save the figure as PNG ---
plt.savefig("duration_distribution_hours.png", dpi=300)
plt.close()

print("📁 Graph saved as: duration_distribution_hours.png")



