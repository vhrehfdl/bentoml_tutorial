from service import KerasTextClassificationService
from binary_pre_training import model, tokenizer

bento_svc = KerasTextClassificationService()
bento_svc.pack('model', model)
bento_svc.pack('tokenizer', tokenizer)

saved_path = bento_svc.save()