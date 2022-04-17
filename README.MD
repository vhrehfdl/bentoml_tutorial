## Requirements
```bash
pip install -r requirements.txt
```


## titanic
scikit learn을 이용한 타이타닉 생존자 분류 모델 만들기. 
### model file save and load
```bash
cd titanic
python train.py
python inference.py
```

### BentoML API Server
```bash
(터미널 A) bentoml serve Titanic:latest
(터미널 B) python request.py
```


## review classification (Naive Bayes)
네이버 영화리뷰 댓글 긍부정 분류 모델 만들기
### model file save and load
```bash
cd review_classification
python NB_train.py
python NB_inference.py
```

### BentoML API Server
```bash
(터미널 A) bentoml serve ReviewClassifier:latest
(터미널 B) python request.py
```


## review classification (KoBERT)
### model file save and load
```bash
python KoBERT_train.py
python KoBERT_inference.py
```

### BentoML API Server
```bash
(터미널 A) bentoml serve ReviewClassifier:latest
(터미널 B) python request.py
```

### MLflow에서 BentoML 버전 체크
```bash
cd review_classification
mlflow ui -h 0.0.0.0 -p 1000
```
