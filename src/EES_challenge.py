import gc
import os
from typing import Dict, Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import class_weight
import scipy.signal as sps
from src.utils.audio_preprocessing_utils import load_wav_file
from src.utils.generators import AudioFixedChunksGenerator, ChunksGenerator_preprocessing
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
        loaded_wav_files[filename]=(wav_file, sample_rate)
    return loaded_wav_files

def test_different_features(feature_types:Tuple[str,...], sequence_max_length:float,
                            window_length:float, num_classes:int=3, path_to_save_results:str='results') -> None:
    path_to_train_data = 'E:\\Databases\\Compare_2021_ESS\\wav\\train\\'
    path_to_train_labels = 'E:\\Databases\\Compare_2021_ESS\\lab\\train.csv'
    path_to_devel_labels = 'E:\\Databases\\Compare_2021_ESS\\lab\\devel.csv'
    path_to_devel_data = 'E:\\Databases\\Compare_2021_ESS\\wav\\dev\\'
    # params
    num_chunks = int(sequence_max_length / window_length)
    label_type = 'sequence_to_one'
    batch_size = 16
    num_mfcc = 128
    subwindow_size=0.2
    subwindow_step=0.1
    # variable for logging
    results=[]

    for feature_type in feature_types:
        # train data
        train_labels = read_labels(path_to_train_labels)
        train_data = load_all_wav_files(path_to_train_data, resample=True)
        train_generator = ChunksGenerator_preprocessing(sequence_max_length=sequence_max_length,
                                                        window_length=window_length,
                                                        data=train_data,
                                                        data_preprocessing_mode=feature_type,
                                                        num_mfcc=num_mfcc,
                                                        labels=train_labels, labels_type=label_type,
                                                        batch_size=batch_size,
                                                        normalization=True, one_hot_labeling=True,
                                                        num_classes=num_classes,
                                                        subwindow_size=subwindow_size, subwindow_step=subwindow_step,
                                                        precutting=False, precutting_window_size=None,
                                                        precutting_window_step=None)

        del train_data
        gc.collect()

        # validation data
        devel_labels = read_labels(path_to_devel_labels)
        val_data = load_all_wav_files(path_to_devel_data, resample=True)
        devel_generator = ChunksGenerator_preprocessing(sequence_max_length=sequence_max_length,
                                                        window_length=window_length,
                                                        data=val_data,
                                                        data_preprocessing_mode=feature_type,
                                                        num_mfcc=num_mfcc,
                                                        labels=devel_labels, labels_type=label_type,
                                                        batch_size=batch_size,
                                                        normalization=True, one_hot_labeling=True,
                                                        num_classes=num_classes,
                                                        subwindow_size=subwindow_size, subwindow_step=subwindow_step,
                                                        precutting=False, precutting_window_size=None,
                                                        precutting_window_step=None)
        del val_data
        gc.collect()

        # build model
        input_shape = (num_chunks,) + train_generator.__get_features_shape__()[-2:]
        model = chunk_based_rnn_model(input_shape=input_shape, num_output_neurons=num_classes,
                                      neurons_on_layer=(128, 128), rnn_type='LSTM',
                                      need_regularization=True, dropout=True)
        model.summary()
        model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Recall()])
        # compute class weights
        train_labels = pd.read_csv(path_to_train_labels)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(train_labels['label'].values),
                                                          train_labels['label'].values)
        class_weights = {i: class_weights[i] for i in range(num_classes)}
        # train
        hist=model.fit(train_generator, epochs=50, validation_data=devel_generator, class_weight=class_weights)
        # collect the best reached score
        best_score = max(hist.history['val_recall'])
        results.append((feature_type, best_score))
        print('feature_type:%s, max_val_recall:%f'%(feature_type, best_score))
        # clear RAM
        del model
        del hist
        gc.collect()
        tf.keras.backend.clear_session()
    print(results)
    pd.DataFrame(results, columns=['feature_type','val_recall']).to_csv(path_to_save_results, index=False)



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
    data_preprocessing_mode = 'HLD'
    label_type = 'sequence_to_one'
    batch_size = 8
    num_mfcc=128

    test_different_features(feature_types=('MFCC',),
                            sequence_max_length=sequence_max_length, window_length=window_length)


