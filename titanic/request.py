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

response = requests.post('http://14.49.44.212:1000/predict', headers=headers, json=json_data)
print(response.text)
