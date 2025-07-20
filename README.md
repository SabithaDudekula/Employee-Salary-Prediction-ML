# 🧠 Employee Salary Prediction Using Machine Learning

## 📌 Objective
To develop a simple machine learning model that predicts salary using historical employee data.

## 📂 Project Files
- `Salary_Data.csv` → The dataset used for training
- `salary_predict.py` → Python script to train and test the model
- `model.pkl` → Saved machine learning model
- `README.md` → Project documentation

## 🛠️ Tools & Libraries Used
- Python
- Pandas
- NumPy
- scikit-learn
- Pickle

## 🔍 How It Works
1. Load the dataset
2. Train a Linear Regression model using scikit-learn
3. Save the trained model as `model.pkl`
4. Use it to predict salary for any years of experience

## 🧪 Sample Prediction Output
```
Predicted Salary for 5 years exp: ₹78342.25
```



## 💡 Usage
You can reuse this model to predict salaries by changing the input value in the script:

```python
exp = [[7]]  # for 7 years of experience
prediction = model.predict(exp)
print(f"Predicted Salary for 7 years exp: ₹{prediction[0]}")

```

