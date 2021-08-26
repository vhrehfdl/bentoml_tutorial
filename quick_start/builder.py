from service import IrisClassifierService
from model import clf

iris_classifier_service = IrisClassifierService()
iris_classifier_service.pack('model', clf)

saved_path = iris_classifier_service.save()