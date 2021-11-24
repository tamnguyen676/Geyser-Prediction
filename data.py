import pandas as pd
import numpy as np
import datetime

import tensorflow as tf

from pytz import timezone

GITHUB_URL = 'https://raw.githubusercontent.com/tamnguyen676/Geyser-Prediction/master'
DRIVE_URL = '/content/drive/MyDrive'


class Data:
    def __init__(self, csv_name, features, train_percent, val_percent, test_percent, batch_size):
        if train_percent + val_percent + test_percent != 1:
            raise ValueError('Arguments must add up to 1')

        self.batch_size = batch_size
        path = f'{GITHUB_URL}/csv/{csv_name}.csv'
        print(path)
        self.full_dataframe = pd.read_csv(path, parse_dates=['datetime'])

        self.dataset_size = len(self.full_dataframe)

        self.train_size = int(train_percent * self.dataset_size)
        self.val_size = int(val_percent * self.dataset_size)
        self.test_size = int(test_percent * self.dataset_size)

    def split_train_val_test(self, full_dataset):
        full_dataset = full_dataset.shuffle(buffer_size=self.dataset_size, seed=0, reshuffle_each_iteration=False)
        self.train_dataset = full_dataset.take(self.train_size).prefetch(1000).batch(self.batch_size)
        test_dataset = full_dataset.skip(self.train_size)
        self.val_dataset = test_dataset.take(self.val_size).prefetch(1000).batch(self.batch_size)
        self.test_dataset = test_dataset.skip(self.test_size).prefetch(1000).batch(self.batch_size)


# Starts with an eruption
# Each element is one time interval long (5 minutes by default)
# Duration: 320 0 0 0 0 0 0 0 0 0 0 0 150 0 0   0's if on eruption in that time window else duration of eruption (s)
# Offset:   5   0 0 0 0 0 0 0 0 0 0 0 45 0 0    offset in seconds for time window of eruption
# Height:   54  0 0 0 0 0 0 0 0 0 0 0 43 0 0    Height of eruption
class NpsDataSequential(Data):
    def __init__(self, features=None, train_percent=.6, val_percent=.2,
                 test_percent=.2, batch_size=20, interval=5):
        super().__init__('old_faithful_nps_sequential', features, train_percent, val_percent, test_percent, batch_size)

        self.interval = interval
        self.sequence_length = 120 // interval

        dict_output_types = {'duration': tf.float32, 'height': tf.float32, 'offset': tf.float32}

        tf_shape = tf.TensorShape([self.sequence_length])
        dict_output_shapes = {'duration': tf_shape, 'height': tf_shape, 'offset': tf_shape}

        full_dataset = tf.data.Dataset.from_generator(self._generator,
                                                      output_types=(tf.float32, tf.float32),
                                                      output_shapes=(tf.TensorShape([3 * self.sequence_length]), tf.TensorShape([])))
        self.split_train_val_test(full_dataset)

    def _generator(self):
        df = self.full_dataframe
        # Generator (to act as dataset)
        for df_index in range(len(df) - 1):
            previous_eruption, current_eruption = df.iloc[df_index], df.iloc[df_index + 1]
            # last index is time_delta
            if (current_eruption['datetime'] - previous_eruption['datetime']).seconds <= 2 * 60 * 60:
                yield self._create_sequences(previous_eruption, current_eruption), current_eruption[
                    'time_to_next_eruption']

    def _create_sequences(self, previous_eruption, current_eruption):
        features = {
            'duration': np.zeros(shape=(self.sequence_length,)),
            'height': np.zeros(shape=(self.sequence_length,)),
            'offset': np.zeros(shape=(self.sequence_length,))
        }

        start_time = previous_eruption['datetime']
        tmp_time = datetime.datetime(year=start_time.year, month=start_time.month, day=start_time.day,
                                     hour=start_time.hour, minute=start_time.minute - start_time.minute % self.interval,
                                     second=0, tzinfo=timezone('UTC'))

        for i in range(self.sequence_length):
            for eruption in (previous_eruption, current_eruption):
                if tmp_time <= eruption['datetime'] < datetime.timedelta(minutes=self.interval) + tmp_time:
                    time = eruption['datetime']
                    features['duration'][i] = eruption['duration']
                    features['height'][i] = eruption['height']
                    features['offset'][i] = time.minute - time.minute % self.interval + time.second

            tmp_time = datetime.timedelta(minutes=self.interval) + tmp_time
        return np.hstack((features['duration'], features['height'], features['offset']))


class NpsData(Data):
    def __init__(self, features=None, train_percent=.6, val_percent=.2,
                 test_percent=.2, batch_size=20):
        super().__init__('old_faithful_nps', features, train_percent, val_percent, test_percent, batch_size)
        if features is None:
            features = ['duration', 'time_since_last_eruption', 'height', 'interpolated_height']
        labels = self.full_dataframe.pop('time_to_next_eruption')
        features = self.full_dataframe[features]
        full_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        self.split_train_val_test(full_dataset)



