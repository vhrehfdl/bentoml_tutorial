# builder.py

# import the IrisClassifier class defined above
from service import PytorchTextClassifier
from model import basic_model

# Create a iris classifier service instance
iris_classifier_service = PytorchTextClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('BasicModel', basic_model)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()