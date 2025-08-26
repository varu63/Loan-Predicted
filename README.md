Loan Prediction System

A Machine Learning + Flask web application that predicts whether a loan application will be approved or rejected based on applicant details such as income, education, employment status, credit history, and property area.

🚀 Features

Data preprocessing and training on loan dataset

Logistic Regression model for loan approval prediction

Web interface built with Flask + HTML (templates)

Simple form to collect applicant details

Real-time prediction with trained model

🛠️ Tech Stack

Python 3.x

Flask – Web framework

pandas, numpy – Data processing

scikit-learn – Machine learning (Logistic Regression)

joblib – Model persistence

HTML / CSS – Frontend (templates)

📂 Project Structure
Loan-Predicted/
│── app.py              # Flask app for prediction  
│── main.py             # Model training script  
│── loan_prediction_model.pkl  # Saved ML model (generated after training)  
│── templates/          # HTML templates  
│    ├── index.html     # Input form  
│    ├── result.html    # Prediction output page  
│── train.csv           # Training dataset (if included)  
│── test.csv            # Testing dataset (if included)  
│── README.md           # Project documentation  

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/varu63/Loan-Predicted.git
cd Loan-Predicted


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Mac/Linux  
venv\Scripts\activate      # On Windows  


Install dependencies

pip install -r requirements.txt


(If requirements.txt is not present, install manually:)

pip install flask pandas scikit-learn joblib


Run the app

python app.py


The application will start at: http://127.0.0.1:5000/

🎯 Usage

Open the app in your browser.

Enter applicant details in the form.

Submit to check if the loan is Approved / Not Approved.

📊 Model Details

Algorithm Used: Logistic Regression

Target Variable: Loan_Status (Approved / Rejected)

Input Features:

Gender, Married, Dependents, Education, Self_Employed

ApplicantIncome, CoapplicantIncome

LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

🤝 Contribution

Feel free to fork this repo, raise issues, or submit PRs to improve the project.
