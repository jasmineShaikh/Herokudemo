import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'satisfaction_level':0.58,'last_evaluation':0.38,'number_project':3,'average_montly_hours':157,'time_spend_company':3,'Work_accident':0,'promotion_last_5years':0,'sales':8,'salary':2})

print(r.json())