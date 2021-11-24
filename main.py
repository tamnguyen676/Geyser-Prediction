from data import NpsDataSequential
from trainer import ModelTrainer
from models import DefaultModel

data = NpsDataSequential()
model, history = ModelTrainer.train_model(data.train_dataset, data.val_dataset,
                                          DefaultModel.get_model(data.train_dataset), epochs=150)
