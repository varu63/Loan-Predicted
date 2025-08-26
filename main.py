import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , classification_report,mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression 
import joblib

# Load dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# print(train_data.shape)
# print(train_data.isnull().sum())

# Preprocessing function
def preprocess(data):
    data.fillna({
    'Gender':'Male',
    'Married':'Yes',
    'Dependents':'0',
    'Self_Employed':'No',
    'LoanAmount':data['LoanAmount'].median()if 'LoanAmount' in data else 0,
    'Loan_Amount_Term':360,
    'Credit_History':1.0
    },inplace=True)
    if "Dependents" in data.columns:
        data["Dependents"] = data["Dependents"].astype(str).str.replace("+", "", regex=False)
        data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce")

# Encode the categorical data 
    categorical_colums = ['Gender','Married','Education','Self_Employed','Property_Area']
    for col in categorical_colums:
        le = LabelEncoder()
        data[col] = data[col].astype(str).fillna("Unknown")
        data[col] = le.fit_transform(data[col])
    return data



# Preprocess data
train_data = preprocess(train_data)
test_data = preprocess(test_data)
test_data =test_data.drop(columns=['Loan_ID'])
# Split features & labels
X = train_data.drop(columns=['Loan_ID','Loan_Status'])
y=train_data['Loan_Status']


# Train-test split
X_train ,X_val ,y_train ,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train,y_train)


# Predictions
y_pred = model.predict(X_train)

# Evaluation
print("Accuracy:", accuracy_score(y_train, y_pred))
print(classification_report(y_train, y_pred))

# ---------------- SAVE MODEL ----------------
# joblib.dump(model, "loan_prediction_model.pkl")
# print("Model saved as loan_prediction_model.pkl")

predict = model.predict(test_data)
print(predict)



