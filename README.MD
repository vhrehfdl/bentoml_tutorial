## Requirements
```bash
pip install -r requirements.txt
```

## Example
### Case1
```bash
cd keras
cd binary_pre_training
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove*.zip
python builder.py
bentoml serve KerasClassification --port 1000
```

### Case2
```bash
cd keras
cd multi_self_training
python builder.py
bentoml serve KerasClassification --port 1000
```
