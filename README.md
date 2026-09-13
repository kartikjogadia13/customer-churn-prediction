# Customer Churn Prediction

## Project goal

This project predicts whether a telecom customer is likely to leave a service. It uses basic customer details such as age, monthly charge, time with the company, contract type, and internet service.

## Tools used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Streamlit (optional web interface)

## How it works

1. Read customer data from `data/customer_churn.csv`.
2. Convert text categories such as contract type into numbers using one-hot encoding.
3. Train a Logistic Regression classification model.
4. Test the model and display its accuracy.
5. Predict churn for a new customer in the Streamlit interface.

## Run in VS Code

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

To open the simple interface:

```powershell
streamlit run streamlit_app.py
```

## Dataset note

The included CSV is a small practice dataset created for demonstration. For a larger version of this project, replace it with a public telecom churn dataset while keeping the same column names.
