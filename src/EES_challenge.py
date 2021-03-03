import os
from typing import Dict

import tensorflow as tf
import pandas as pd
import numpy as np

from src.utils.generators import AudioFixedChunksGenerator
from src.utils.tf_utils import chunk_based_rnn_model


def read_labels(path:str) -> Dict[str, np.ndarray]:
    labels=pd.read_csv(path)
    result_dict={}
    for index, row in labels.iterrows():
        result_dict[row['filename']]=np.array(row['label']).reshape((1,1))
    return result_dict





if __name__ == '__main__':
    path_to_train_data='D:\\Databases\\Compare_2021_ESS\\wav\\train\\'
    path_to_train_labels='D:\\Databases\\Compare_2021_ESS\\lab\\train.csv'
    path_to_devel_labels='D:\\Databases\\Compare_2021_ESS\\lab\\devel.csv'
    path_to_devel_data = 'D:\\Databases\\Compare_2021_ESS\\wav\\dev\\'
    # params
    sequence_max_length = 12
    window_length = 0.5
    num_chunks = int(sequence_max_length / window_length)
    num_classes=3
    data_preprocessing_mode = 'LLD'
    label_type = 'sequence_to_one'
    batch_size = 4

    # train data
    train_labels=read_labels(path_to_train_labels)
    train_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='path',
                                          load_path=path_to_train_data,
                                          data_preprocessing_mode=data_preprocessing_mode,
                                          labels=train_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=3)
    # validation data
    devel_labels=read_labels(path_to_devel_labels)
    devel_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='path',
                                          load_path=path_to_devel_data,
                                          data_preprocessing_mode=data_preprocessing_mode,
                                          labels=devel_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=3)

    model=chunk_based_rnn_model(input_shape=(num_chunks,46,65), num_output_neurons=num_classes,
    neurons_on_layer = (256, 256), rnn_type = 'LSTM')
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Recall()])

    model.fit(train_generator, epochs=10, validation_data=devel_generator)
    '''
    for x,y in generator:
        loss=model.train_on_batch(x, y)
        print(loss)'''

