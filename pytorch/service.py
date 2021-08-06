from bentoml import env, artifacts, api, BentoService
from bentoml.frameworks.pytorch import PytorchModelArtifact
from bentoml.adapters import StringInput


@env(infer_pip_packages=True)
@artifacts([PytorchModelArtifact('BasicModel')])
class PytorchTextClassifier(BentoService):
    
    @api(input=StringInput(), batch=True)
    def predict(self, txt):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(txt)