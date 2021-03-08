import os
from typing import Dict

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import class_weight

from src.utils.generators import AudioFixedChunksGenerator
from src.utils.tf_utils import chunk_based_rnn_model


def read_labels(path:str) -> Dict[str, np.ndarray]:
    labels=pd.read_csv(path)
    result_dict={}
    for index, row in labels.iterrows():
        result_dict[row['filename']]=np.array(row['label']).reshape((1,1))
    return result_dict





if __name__ == '__main__':
    path_to_train_data='E:\\Databases\\Compare_2021_ESS\\wav\\train\\'
    path_to_train_labels='E:\\Databases\\Compare_2021_ESS\\lab\\train.csv'
    path_to_devel_labels='E:\\Databases\\Compare_2021_ESS\\lab\\devel.csv'
    path_to_devel_data = 'E:\\Databases\\Compare_2021_ESS\\wav\\dev\\'
    # params
    sequence_max_length = 12
    window_length = 0.5
    num_chunks = int(sequence_max_length / window_length)
    num_classes=3
    data_preprocessing_mode = 'EGEMAPS'
    label_type = 'sequence_to_one'
    batch_size = 16
    num_mfcc=128

    # train data
    train_labels=read_labels(path_to_train_labels)
    train_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='path',
                                          load_path=path_to_train_data,
                                          data_preprocessing_mode=data_preprocessing_mode, num_mfcc=num_mfcc,
                                          labels=train_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=num_classes,
                                          subwindow_size=0.25, subwindow_step=0.1)

    # validation data
    devel_labels=read_labels(path_to_devel_labels)
    devel_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='path',
                                          load_path=path_to_devel_data,
                                          data_preprocessing_mode=data_preprocessing_mode,num_mfcc=num_mfcc,
                                          labels=devel_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=num_classes)

    model=chunk_based_rnn_model(input_shape=(num_chunks,32,128), num_output_neurons=num_classes,
    neurons_on_layer = (256, 256), rnn_type = 'LSTM',
                                need_regularization=True, dropout=True)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Recall()])

    train_labels = pd.read_csv(path_to_train_labels)

    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(train_labels['label'].values),
                                                      train_labels['label'].values)

    class_weights = {i: class_weights[i] for i in range(num_classes)}

    print(class_weights)

    model.fit(train_generator, epochs=50, validation_data=devel_generator, class_weight=class_weights)


