import pickle
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Type


def pos_tagging(sentences: str):
    tagger = Okt()
    pos_sentences = [" ".join(tagger.nouns(sentence)) for sentence in sentences]
    return pos_sentences


def custom_tokenizer(text: str):
    return text.split(" ")


if __name__ == "__main__":
    # Directory Setting
    model_dir = "model_nb.pickle"
    tokenizer_dir = "tokenizer_nb.pickle"

    # Flow
    input_text = ["시간버렷다", "와 감동스러운 영화 그 자체~!", "영화 대박 엄청 재미있어요"]

    tokenizer = pickle.load(open(tokenizer_dir, 'rb'))
    model = pickle.load(open(model_dir, 'rb'))

    input_text = pos_tagging(input_text)
    input_text = tokenizer.transform(input_text)
    
    print("pred_y : ", model.predict(input_text))