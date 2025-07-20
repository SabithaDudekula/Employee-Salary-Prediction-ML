import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("Salary_Data.csv")
X = data[['YearsExperience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Optional: Predict for 5 years experience
exp = [[5]]
predicted_salary = model.predict(exp)
print(f"Predicted Salary for 5 years exp: ₹{predicted_salary[0]:.2f}")
