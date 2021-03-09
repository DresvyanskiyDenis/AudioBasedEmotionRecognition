import gc
import os
from typing import Dict, Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import scipy.signal as sps
from src.utils.audio_preprocessing_utils import load_wav_file
from src.utils.generators import AudioFixedChunksGenerator
from src.utils.tf_utils import chunk_based_rnn_model


def read_labels(path:str) -> Dict[str, np.ndarray]:
    labels=pd.read_csv(path)
    result_dict={}
    for index, row in labels.iterrows():
        result_dict[row['filename']]=np.array(row['label']).reshape((1,1))
    return result_dict

def load_all_wav_files(path:str, resample:bool=False, new_sample_rate:int=16000)-> Dict[str,Tuple[np.ndarray, int]]:
    loaded_wav_files={}
    filenames=os.listdir(path)
    for filename in filenames:
        sample_rate,wav_file=load_wav_file(os.path.join(path,filename))
        # Resample data
        if resample and sample_rate!=new_sample_rate:
            number_of_samples = round(len(wav_file) * new_sample_rate / sample_rate)
            wav_file = sps.resample(wav_file, number_of_samples)
            sample_rate=new_sample_rate

        if len(wav_file.shape)<2: wav_file=wav_file[..., np.newaxis]
        loaded_wav_files[filename.split(".")[0]]=(wav_file, sample_rate)
    return loaded_wav_files


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
    batch_size = 16
    num_mfcc=128

    # train data
    train_labels=read_labels(path_to_train_labels)
    train_data=load_all_wav_files(path_to_train_data, resample=True)
    train_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='data',
                                          data=train_data,
                                          data_preprocessing_mode=data_preprocessing_mode, num_mfcc=num_mfcc,
                                          labels=train_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=num_classes,
                                          subwindow_size=0.1, subwindow_step=0.05)
    for x,y in train_generator:
        print(x,y)

    del train_data
    gc.collect()
    # validation data
    devel_labels=read_labels(path_to_devel_labels)
    val_data=load_all_wav_files(path_to_devel_data, resample=True)
    devel_generator = AudioFixedChunksGenerator(sequence_max_length=sequence_max_length, window_length=window_length,
                                          load_mode='data',
                                          data=val_data,
                                          data_preprocessing_mode=data_preprocessing_mode,num_mfcc=num_mfcc,
                                          labels=devel_labels, labels_type='sequence_to_one', batch_size=batch_size,
                                          normalization=True, one_hot_labeling=True, num_classes=num_classes)
    del val_data
    gc.collect()
    model=chunk_based_rnn_model(input_shape=(num_chunks,22,260), num_output_neurons=num_classes,
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


