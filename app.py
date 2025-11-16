from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    CreditScore = float(request.form['CreditScore'])
    Age = float(request.form['Age'])
    Tenure = float(request.form['Tenure'])
    Balance = float(request.form['Balance'])
    NumOfProducts = float(request.form['NumOfProducts'])
    HasCrCard = float(request.form['HasCrCard'])
    IsActiveMember = float(request.form['IsActiveMember'])
    EstimatedSalary = float(request.form['EstimatedSalary'])
    Geography_Germany = float(request.form['Geography_Germany'])
    Geography_Spain = float(request.form['Geography_Spain'])
    Gender_Male = float(request.form['Gender_Male'])

    features = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts,
                          HasCrCard, IsActiveMember, EstimatedSalary,
                          Geography_Germany, Geography_Spain, Gender_Male]])

    final_features = scaler.transform(features)

    prediction = model.predict(final_features)[0]

    if prediction == 1:
        result = "This customer will CHURN ❌"
    else:
        result = "This customer will NOT churn ✅"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
