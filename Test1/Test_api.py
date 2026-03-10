import requests

url = "http://127.0.0.1:8000/predict"
sample_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

response = requests.post(url, json=sample_data)

if response.status_code == 200:
    print("Success!")
    print(f"Prediction: {response.json()['prediction']}")
    print(f"Model used: {response.json()['model_version']}")
else:
    print(f"Failed with status: {response.status_code}")
    print(response.text)