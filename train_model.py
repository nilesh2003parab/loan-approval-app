import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/loan_data.csv")

df = df.ffill()


le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
with open("model/loan_model.pkl", "wb") as f:
    pickle.dump(model, f)
