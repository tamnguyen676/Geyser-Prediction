import tensorflow as tf
from tensorflow import keras
from data import NpsDataSequential
from model import DefaultModel, SequentialModel


class ModelTrainer:
    @staticmethod
    def train_model(train_dataset, val_dataset, model=None, loss='mean_absolute_error', epochs=30,
                    learning_rate=.001, verbose=True):
        if model is None:
            model = DefaultModel.get_model(train_dataset)

        model.compile(loss=loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate),
                      metrics=[keras.metrics.RootMeanSquaredError()])

        history = model.fit(train_dataset, validation_data=val_dataset,
                            verbose=verbose, epochs=epochs)

        return model, history
