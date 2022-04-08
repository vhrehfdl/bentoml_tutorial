import torch

from typing import List
from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.service.artifacts.common import PickleArtifact
from bentoml.adapters import JsonInput
from bentoml.types import JsonSerializable
from BERT import BERTDataset, BERTClassifier


@artifacts([PytorchModelArtifact('model'), PickleArtifact('tokenizer'), PickleArtifact('max_len'), 
            PickleArtifact('BERTDataset'), PickleArtifact('label_enc')])
class ReviewClassifier(BentoService):
    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_jsons: List[JsonSerializable]):
        max_len = self.artifacts.max_len
        device = torch.device("cuda")
        input_texts = parsed_jsons[0]['text']
        
        data_test = self.artifacts.BERTDataset(input_texts, [0]*len(input_texts), self.artifacts.tokenizer, max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False)

        pred_y = []
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            output = self.artifacts.model(token_ids.long().to(device), valid_length, segment_ids.long().to(device))
            _, output = torch.max(output, 1)
            output = output.tolist()
            output = self.artifacts.label_enc.inverse_transform(output)
            pred_y.append(output)
            
        return [pred_y]