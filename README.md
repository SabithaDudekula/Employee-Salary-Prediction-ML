
# 🧠 Employee Salary Prediction Using Machine Learning

## 📌 Objective
To develop a simple machine learning model that predicts salary using historical employee data.

## 📂 Project Files
- `Salary_Data.csv` → The dataset used for training  
- `salary_predict.py` → Python script to train and test the model  
- `model.pkl` → Saved machine learning model  
- `README.md` → Project documentation  

## 🛠️ Tools & Libraries Used

### 💻 System Requirements
- `Python`
- `Jupyter Notebook` / `VS Code`
- `Streamlit` – for Web App Development

### 📦 Libraries Required to Build the Model
- `pandas`, `numpy` – data manipulation
- `scikit-learn` – model building & evaluation
- `lightgbm` – gradient boosting classifier
- `matplotlib`, `seaborn` – visualizations
- `streamlit` – interactive web interface

## 🔍 How It Works
1. Load dataset and clean missing values.
2. Chose LightGBM, a gradient boosting framework known for performance and speed.
3. Train the LightGBM classifier. Evaluate using accuracy, pressision, and confusion matrix.
4. Developed an interactive frontend using Streamlit where users can input their profiles to predict salary class.
5. Hosted locally and optionally on platform like Streamlit.

## 🧪 Sample Prediction Output

```

Predicted Salary for 5 years exp: ₹78342.25

````

## 💡 Usage
You can reuse this model to predict salaries by changing the input value in the script:

```python
exp = [[7]]  # for 7 years of experience
prediction = model.predict(exp)
print(f"Predicted Salary for 7 years exp: ₹{prediction[0]}")
````

## 🚀 Setup & Installation

1. Clone this repository or download the files
2. Install the required libraries using:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python salary_predict.py
```

## 📊 Model Evaluation Results

### 📌 Salary Distribution by Experience
![Result 1](result1.png)

### 📌 Predicted vs Actual Salaries
![Result 2](result2.png)

### 📌 Feature Importance
![Result 3](result3.png)

### 📌 Model Performance Metrics
![Result 4](result4.png)


```

---

