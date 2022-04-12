import requests

headers = {
    'Content-Type': 'application/json',
}

json_data = {
    'columns': ['Pclass', 'Sex', 'Fare', 'SibSp', 'Parch'],
    'data': [
        [3, 0, 1, 1, 2],
        [2, 3, 2, 2, 2],
    ],
}

response = requests.post('http://10.20.81.77:5000/predict', headers=headers, json=json_data)
print(response.text)
