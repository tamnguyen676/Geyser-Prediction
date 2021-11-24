from data import NpsDataSequential
from trainer import ModelTrainer
from model import DefaultModel

data = NpsDataSequential()
model, history = ModelTrainer.train_model(data.train_dataset, data.val_dataset,
                                          DefaultNetwork.get_model(data.train_dataset), epochs=150)
