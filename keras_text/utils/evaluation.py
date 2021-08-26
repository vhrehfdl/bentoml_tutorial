from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class Evaluation:
    def __init__(self, model, test_x, test_y):
        self.model = model
        self.test_x = test_x
        self.test_y = test_y

    def eval_classification(self, data_type):
        predictions = self.model.predict(self.test_x)

        if data_type == "binary":
            y_pred = (predictions > 0.5)
        elif data_type == "multi":
            y_pred = predictions.argmax(axis=-1)

        accuracy = accuracy_score(self.test_y, y_pred)
        cf_matrix = confusion_matrix(self.test_y, y_pred)
        report = classification_report(self.test_y, y_pred)

        return accuracy, cf_matrix, report

    def eval_classification_bert(self, data_type):
        predictions = self.model.predict(self.test_x)

        if data_type == "binary":
            y_pred = (predictions > 0.5)
        elif data_type == "multi":
            y_pred = predictions.argmax(axis=-1)

        accuracy = accuracy_score(self.test_y, y_pred)
        cf_matrix = confusion_matrix(self.test_y, y_pred)
        report = classification_report(self.test_y, y_pred)

        return accuracy, cf_matrix, report
