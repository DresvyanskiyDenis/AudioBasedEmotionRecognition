#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO: add file description and functions list
"""
import contextlib
import wave
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os
import scipy.signal as sps
import tensorflow as tf

__author__ = "Denis Dresvyanskiy"
__copyright__ = "Copyright 2021"
__credits__ = ["Denis Dresvyanskiy"]
__maintainer__ = "Denis Dresvyanskiy"
__email__ = "denis.dresvyanskiy@uni-ulm.de"

from sklearn.metrics import recall_score

from src.utils.audio_preprocessing_utils import load_wav_file
from src.utils.generator_loader import FixedChunksGenerator_loader
from src.utils.sequence_to_one_models import chunk_based_1d_cnn_attention_model

"""
Angry		(A)
Sad		    (S)
Happy		(H)
Surprise	(U)
Fear		(F)
Disgust		(D)
Contempt	(C)
Neutral		(N)
Other		(O)
No agreement(X)
"""
emotion_to_int_mappping={
    'A': 0,
    'S': 1,
    'H': 2,
    'U': 3,
    'F': 4,
    'D': 5,
    'C': -1,
    'N': 6,
    'O': -1,
    'X': -1
}

def load_and_preprocess_labels_MSP_podcast(path:str)->Dict[str, np.ndarray]:
    """Loads and preprocesses labels of MSP-podcast dataset. The all labels are contained in one .txt file.
       The typical format of line in file is
       MSP-PODCAST_0001_0008.wav; N; A:float_value; V:float_value; D:float_value;
       There are also some other lines, which should be ignored.

    :param path: str
            path to the file with labels
    :return: Dict[str, int]
            loaded labels in format Dict[filename->emotion_category]
    """
    labels={}
    # open file
    with open(path) as file_object:
        for line in file_object:
            # check if we read the needed line. It always shoul starts from 'MSP-PODCAST'
            if line[:11]=='MSP-PODCAST':
                # split line by ';' and throw out last element of array, because it will be always empty string
                splitted_line=line.split(';')[:-1]
                # save extracted label and filename
                filename=splitted_line[0]
                class_category=splitted_line[1].strip()
                labels[filename]=np.array(emotion_to_int_mappping[class_category]).reshape((-1,))
    return labels

def delete_instances_with_class(labels:Dict[str, np.ndarray], class_to_delete:int=-1)-> Dict[str, np.ndarray]:
    # TODO: add description
        keys_to_delete=[]
        for key, item in labels.items():
            if item==class_to_delete:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            labels.pop(key)
        return labels

def get_train_dev_separation_in_dict(path:str)-> Dict[str, List[str]]:
    #TODO: write description
    labels_partition_dict={}
    partition_df=pd.read_csv(path, sep=';', header=None)
    labels_partition_dict['train']=partition_df[partition_df.iloc[:,0]=='Train'].iloc[:,1].to_list()
    labels_partition_dict['dev'] = partition_df[partition_df.iloc[:,0]=='Validation'].iloc[:,1].to_list()
    return labels_partition_dict

def get_labels_according_to_filenames(labels:Dict[str, np.ndarray],
                                                  filenames:List[str])-> Dict[str, np.ndarray]:
    # TODO: write description
    extracted_labels=dict((filename.strip(), labels[filename.strip()]) for filename in filenames)
    return extracted_labels

def custom_recall_validation_with_generator(generator:FixedChunksGenerator_loader, model:tf.keras.Model)->float:
    total_predictions=np.zeros((0,))
    total_ground_truth=np.zeros((0,))
    for x,y in generator:
        predictions=model.predict(x)
        predictions=predictions.argmax(axis=-1).reshape((-1,))
        total_predictions=np.append(total_predictions,predictions)
        total_ground_truth=np.append(total_ground_truth,y.argmax(axis=-1).reshape((-1,)))
    return recall_score(total_ground_truth, total_predictions,average='macro')

class validation_callback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator:FixedChunksGenerator_loader):
        super(validation_callback, self).__init__()
        # best_weights to store the weights at which the minimum UAR occurs.
        self.best_weights = None
        # generator to iterate on it on every end of epoch
        self.val_generator=val_generator

    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        current_recall=custom_recall_validation_with_generator(self.val_generator, self.model)
        print('current_recall:', current_recall)
        if np.greater(current_recall, self.best):
            self.best = current_recall
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_weights)



def main():
    path_to_labels = 'E:\\Databases\\MSP_podcast\\labels\\Labels.txt'
    path_to_data = 'E:\\Databases\\MSP_podcast\\Audios'
    path_to_labels_partition='E:\\Databases\\MSP_podcast\\Partitions.txt'
    path_to_save_model_weights='best_model'
    one_hot_labeling = True
    num_classes = 7
    normalization = False
    data_preprocessing_mode = 'raw'
    num_mfcc = 128
    sequence_max_length = 14
    window_length = 1
    batch_size=5
    # load and preprocess labels
    labels = load_and_preprocess_labels_MSP_podcast(path_to_labels)
    # get labels partition (on train and dev) and split labels on train and dev
    labels_partition=get_train_dev_separation_in_dict(path_to_labels_partition)
    train_labels= get_labels_according_to_filenames(labels, labels_partition['train'])
    dev_labels=get_labels_according_to_filenames(labels, labels_partition['dev'])

    # delete instance with inappropriate class (-1)
    train_labels = delete_instances_with_class(train_labels, class_to_delete=-1)
    dev_labels = delete_instances_with_class(dev_labels, class_to_delete=-1)
    # create generator-loader
    train_generator=FixedChunksGenerator_loader(sequence_max_length=sequence_max_length, window_length=window_length,
                                                load_path= path_to_data, resample=16000,
                 data_preprocessing_mode= data_preprocessing_mode, num_mfcc = num_mfcc,
                 labels = train_labels, labels_type= 'sequence_to_one', batch_size=batch_size ,
                 normalization= normalization, one_hot_labeling = one_hot_labeling,
                 num_classes = num_classes)
    # define dev generato and callback to evaluate model performance at the end of epoch
    dev_generator=FixedChunksGenerator_loader(sequence_max_length=sequence_max_length, window_length=window_length,
                                              load_path= path_to_data,resample=16000,
                 data_preprocessing_mode= data_preprocessing_mode, num_mfcc = num_mfcc,
                 labels = dev_labels, labels_type= 'sequence_to_one', batch_size=batch_size ,
                 normalization= normalization, one_hot_labeling = one_hot_labeling,
                 num_classes = num_classes)
    dev_callback=validation_callback(dev_generator)

    # create and compile model
    input_shape = (14, 16000, 1)
    model = chunk_based_1d_cnn_attention_model(input_shape=input_shape, num_output_neurons=num_classes,
                                               filters_per_layer=(128, 128, 256,256, 256,256),
                                               filter_sizes=(20, 15, 12, 8, 6, 5),
                                               pooling_sizes=(8, 4, 4, 2,2,2),
                                               pooling_step=1,
                                               need_regularization=True,
                                               dropout=True,
                                               dropout_rate=0.3)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001, decay=1e-6), loss='categorical_crossentropy',
                  metrics=[tf.keras.metrics.Recall()])
    model.summary()
    # fit model
    hist = model.fit(train_generator, epochs=1, use_multiprocessing=True, callbacks=[dev_callback])
    # save model. The model will be saved with best weights evaluated on devel set (since the dev_callback will set
    # weights of model at the end of training process)
    if not os.path.exists(path_to_save_model_weights):
        os.mkdir(path_to_save_model_weights)
    model.save_weights(os.path.join(path_to_save_model_weights,'best_model_weights.h5'))




if __name__=='__main__':
    main()