import requests
import json

from requests.api import head

uri = 'http://localhost:5000/predict'

data = {
    "Age": 31.0,
    "DistanceFromHome": 10.0,
    "Education": 1.0,
    "JobLevel": 1.0,
    "MonthlyIncome": 41890.0,
    "NumCompaniesWorked": 0.0,
    "PercentSalaryHike": 23.0,
    "StockOptionLevel": 1.0,
    "TotalWorkingYears": 6.0,
    "TrainingTimesLastYear": 3.0,
    "YearsAtCompany": 5.0,
    "YearsSinceLastPromotion": 1.0,
    "YearsWithCurrManager": 4.0
}

input_data = json.dumps(data)
headers = {"Content-Type": "application/json"}

resp = requests.post(uri, input_data, headers=headers)
print(resp.text)
