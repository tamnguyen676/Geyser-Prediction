from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing

class DefaultModel:
    @staticmethod
    def get_model(dataset):
        num_features = dataset.element_spec[0].shape[-1]
        normalizer = preprocessing.Normalization(input_shape=(num_features,))
        features = dataset.map(lambda x, y: x)
        normalizer.adapt(features)

        model = keras.Sequential([
            normalizer,
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dense(1, activation='linear')
        ])

        return model


class SequentialModel:
    @staticmethod
    def get_model(dataset):
        normalizer = preprocessing.Normalization(input_shape=(1, 24))
        normalizer.adapt(dataset)

        model = keras.Sequential([
            normalizer,
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
            layers.Dense(1, activation='linear')
        ])

        return model