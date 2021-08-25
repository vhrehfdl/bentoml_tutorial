# 1) import the custom BentoService defined above
from keras_text_classification_service import KerasTextClassificationService
from model import model, word_index

# 2) `pack` it with required artifacts
bento_svc = KerasTextClassificationService()
bento_svc.pack('model', model)
bento_svc.pack('word_index', word_index)

# 3) save your BentoSerivce
saved_path = bento_svc.save()