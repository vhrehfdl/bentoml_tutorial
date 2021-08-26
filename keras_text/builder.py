# 1) import the custom BentoService defined above
from service import KerasTextClassificationService
from binary_pre_training import model, tokenizer

# 2) `pack` it with required artifacts
bento_svc = KerasTextClassificationService()
bento_svc.pack('model', model)
bento_svc.pack('tokenizer', tokenizer)

# 3) save your BentoSerivce
saved_path = bento_svc.save()