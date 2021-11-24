class ModelInference:
    @staticmethod
    def evaluate_model(model, test_dataset):
        return model.evaluate(test_dataset)
