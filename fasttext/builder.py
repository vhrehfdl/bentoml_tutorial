import fasttext
model = fasttext.train_supervised(input="cooking.train")

from text_classification import FasttextClassification
svc = FasttextClassification()
svc.pack('model', model)

saved_path = svc.save()