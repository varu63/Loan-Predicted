from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("loan_prediction_model.pkl")

# Home route (form input)
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get input values from form
        gender = int(request.form["Gender"])
        married = int(request.form["Married"])
        dependents = int(request.form["Dependents"])
        education = int(request.form["Education"])
        self_employed = int(request.form["Self_Employed"])
        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_amount_term = float(request.form["Loan_Amount_Term"])
        credit_history = float(request.form["Credit_History"])
        property_area = int(request.form["Property_Area"])

        # Create DataFrame for prediction
        input_data = pd.DataFrame([[gender, married, dependents, education, self_employed,
                                   applicant_income, coapplicant_income, loan_amount,
                                   loan_amount_term, credit_history, property_area]],
                                  columns=['Gender','Married','Dependents','Education','Self_Employed',
                                           'ApplicantIncome','CoapplicantIncome','LoanAmount',
                                           'Loan_Amount_Term','Credit_History','Property_Area'])

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Loan Status Prediction: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
