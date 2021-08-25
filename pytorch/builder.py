from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.nn import util as nn_util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.trainer import Trainer
from overrides import overrides
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.col_name = None
        self.batch_size = None
        self.hidden_size = None
        self.epochs = None
        self.use_gpu = None
        self.lr = None
        self.max_seq_len = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_size=64,
    max_seq_len=300,
    use_gpu=torch.cuda.is_available(),
    col_name=["text", "label"]
)


class LoadData(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len,
                 col_name: List[str] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len
        self.col_name = col_name

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: np.ndarray = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, df) -> Iterator[Instance]:
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row[self.col_name[0]])],
                row[[self.col_name[1]]].values,
            )


class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        output = {"class_logits": class_logits, "loss": self.loss(class_logits, label)}

        return output


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


def tokenizer(x: str):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)[:config.max_seq_len]]


def tonp(tsr):
    return tsr.detach().cpu().numpy()


def load_data(train_dir, test_dir):
    token_indexer = PretrainedBertIndexer(
        pretrained_model="bert-base-uncased",
        max_pieces=config.max_seq_len,
        do_lowercase=True,
     )

    reader = LoadData(tokenizer=tokenizer, token_indexers={"tokens": token_indexer}, col_name=config.col_name)

    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)

    test_y = test_data[config.col_name[1]].tolist()

    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_data = reader.read(train_data)
    val_data = reader.read(val_data)
    test_data = reader.read(test_data)

    return train_data, val_data, test_data, test_y


def pre_processing(train_data):
    vocab = Vocabulary()
    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)
    batch = next(iter(iterator(train_data)))

    bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased", top_layer_only=True)
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)
    bert_dim = word_embeddings.get_output_dim()

    class BertSentencePooler(Seq2VecEncoder):
        def forward(self, embs: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
            return embs[:, 0]

        @overrides
        def get_output_dim(self) -> int:
            return bert_dim

    encoder = BertSentencePooler(vocab)
    model = BaselineModel(word_embeddings, encoder, vocab)

    return model, batch, vocab, iterator


def train(model, batch, iterator, train_data):
    if config.use_gpu:
        model.cuda()

    batch = nn_util.move_to_device(batch, 0 if config.use_gpu else -1)
    loss = model(**batch)["loss"]
    loss.backward()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_data,
        cuda_device=0 if config.use_gpu else -1,
        num_epochs=config.epochs
    )

    trainer.train()

    return model


def evaluate(vocab, model, test_data, test_y):
    seq_iterator = BasicIterator(batch_size=64)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=0 if config.use_gpu else -1)
    test_preds = predictor.predict(test_data)

    y_pred = (test_preds > 0.5)
    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(test_y, y_pred, target_names=["0", "1"]))


# Directory
train_dir = "./data/train.csv"
test_dir = "./data/test.csv"

print("1.Load Data")
train_data, val_data, test_data, test_y = load_data(train_dir, test_dir)

print("2.Build model")
model, batch, vocab, iterator = pre_processing(train_data)

print("3.Train")
model = train(model, batch, iterator, train_data)

print("4.Evaluate")
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(vocab)

predictor = Predictor(model, seq_iterator, cuda_device=0 if config.use_gpu else -1)
# test_preds = predictor.predict(test_data)

# y_pred = (test_preds > 0.5)
# accuracy = accuracy_score(test_y, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(classification_report(test_y, y_pred, target_names=["0", "1"]))


from service import PytorchTextClassifier
svc = PytorchTextClassifier()
svc.pack('net', model)

saved_path = svc.save()