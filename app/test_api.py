import requests

url = "http://127.0.0.1:5000/predict"

sample = {
    "Environmental temperature (C)": 39.0,
    "Heart / Pulse rate (b/min)": 166.0,
    "Sweating": 0.0,
    "Patient temperature": 40.8
}

res = requests.post(url, json=sample)

print("Status:", res.status_code)
print("Response:", res.json())
