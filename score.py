import json
import pickle
import numpy as np

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

data = json.loads("""
{"Age":28.0,"DistanceFromHome":9.0,"Education":4.0,"JobLevel":1.0,"MonthlyIncome":56730.0,"NumCompaniesWorked":5.0,"PercentSalaryHike":14.0,"StockOptionLevel":1.0,"TotalWorkingYears":5.0,"TrainingTimesLastYear":4.0,"YearsAtCompany":3.0,"YearsSinceLastPromotion":2.0,"YearsWithCurrManager":2.0}
""")

data_arr = list(data.values())

X_scaled = scaler.transform(np.array(data_arr).reshape(1, -1))

score = model.predict(X_scaled)[0]

print('result:', score)
