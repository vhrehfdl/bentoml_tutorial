# builder.py

# import the IrisClassifier class defined above
from service import IrisClassifierService
from model import clf

# Create a iris classifier service instance
iris_classifier_service = IrisClassifierService()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()