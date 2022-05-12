import requests

headers = {'Content-Type': 'application/json',}
json_data = {"text":["영화 재미있다 ㅎㅎ", "재미 진짜 없다"]}

for i in range(0, 10):
    response = requests.post('http://10.20.83.220:1000/predict', headers=headers, json=json_data)
    print(response.text)
