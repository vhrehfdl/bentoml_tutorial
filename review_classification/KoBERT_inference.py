import torch
import pickle
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Type
from BERT import BERTDataset, BERTClassifier
from kobert.pytorch_kobert import get_pytorch_kobert_model


if __name__ == "__main__":
    # Setting Directory
    model_dir = "model.pt"
    tokenizer_dir = "tokenizer.pickle"
    label_enc_dir = "label_enc.pickle"

    # Setting Parameter
    device = torch.device("cuda")

    # Hyper Parameter
    max_len = 10
    dr_rate = 0.5

    # Flow
    input_text = ["시간버렷다", "와 감동스러운 영화 그 자체~!", "영화 대박 엄청 재미있어요"]
    
    tokenizer = pickle.load(open(tokenizer_dir, 'rb'))
    label_enc = pickle.load(open(label_enc_dir, 'rb'))

    test_set = BERTDataset(input_text, [0]*len(input_text), tokenizer, max_len, True, False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    kobert_model, vocab = get_pytorch_kobert_model()
    model = BERTClassifier(kobert_model, num_classes=len(label_enc.classes_), dr_rate=dr_rate).to(device)
    model.load_state_dict(torch.load(model_dir))

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
        output = model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
        _, output = torch.max(output, 1)
        output = output.tolist()
        output = label_enc.inverse_transform(output)
        print(output)