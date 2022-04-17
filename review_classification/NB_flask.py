import json
import pickle

from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Type
from flask import request
from flask import Flask, render_template, request, url_for, redirect, session, flash


app = Flask(__name__, static_url_path='/static')


def pos_tagging(sentences: str):
    tagger = Okt()
    pos_sentences = [" ".join(tagger.nouns(sentence)) for sentence in sentences]
    return pos_sentences


def custom_tokenizer(text: str):
    return text.split(" ")


@app.route('/predict', methods=['GET', 'POST'])
def index():
    params = json.loads(request.get_data(), encoding='utf-8')

    # Flow
    input_text = params["text"]
    input_text = pos_tagging(input_text)
    input_text = tokenizer.transform(input_text)
    
    pred_y = model.predict(input_text)
    return json.dumps(pred_y.tolist(), ensure_ascii=False)


if __name__ == '__main__':
    # Directory Setting
    model_dir = "model_nb.pickle"
    tokenizer_dir = "tokenizer_nb.pickle"

    tokenizer = pickle.load(open(tokenizer_dir, 'rb'))
    model = pickle.load(open(model_dir, 'rb'))

    app.run(host='0.0.0.0', port='1001', debug=True)