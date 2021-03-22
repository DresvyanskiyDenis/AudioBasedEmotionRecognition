import gc
import os
from typing import Dict, Tuple

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.utils import class_weight
import scipy.signal as sps
from src.utils.audio_preprocessing_utils import load_wav_file
from src.utils.generators import ChunksGenerator_preprocessing
from src.utils.sequence_to_one_models import chunk_based_rnn_model, chunk_based_rnn_attention_model, \
    chunk_based_1d_cnn_attention_model


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
                            window_length:float, num_classes:int=3, path_to_save_results:str='results',
                            subwindow_size:float=0.2, subwindow_step:float=0.05) -> None:
    path_to_train_data = 'C:\\EES_challenge_data\\wav\\train\\'
    path_to_train_labels = 'C:\\EES_challenge_data\\lab\\train.csv'
    path_to_devel_labels = 'C:\\EES_challenge_data\\lab\\devel.csv'
    path_to_devel_data = 'C:\\EES_challenge_data\\wav\\dev\\'
    # params
    num_chunks = int(sequence_max_length / window_length)
    label_type = 'sequence_to_one'
    batch_size = 4
    num_mfcc = 128
    # check if directory where results will be saved exists
    if not os.path.exists(path_to_save_results):
        os.mkdir(path_to_save_results)
    # variable for logging
    results=[]

    for feature_type in feature_types:
        # params to save best model
        path_to_save_model=os.path.join(path_to_save_results,'%s_subwindow_%f_%f'%(feature_type, subwindow_size, subwindow_step))
        if not os.path.exists(path_to_save_model):
            os.mkdir(path_to_save_model)

        # load train data
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
        #clear RAM
        del train_data
        gc.collect()

        # load validation data
        devel_labels = read_labels(path_to_devel_labels)
        val_data = load_all_wav_files(path_to_devel_data, resample=True)
        devel_generator = ChunksGenerator_preprocessing(sequence_max_length=sequence_max_length,
                                                        window_length=window_length,
                                                        data=val_data,
                                                        data_preprocessing_mode=feature_type,
                                                        num_mfcc=num_mfcc,
                                                        labels=devel_labels, labels_type=label_type,
                                                        batch_size=batch_size,
                                                        normalization=True, normalizer=train_generator.normalizer,
                                                        one_hot_labeling=True,
                                                        num_classes=num_classes,
                                                        subwindow_size=subwindow_size, subwindow_step=subwindow_step,
                                                        precutting=False, precutting_window_size=None,
                                                        precutting_window_step=None)
        # clear RAM
        del val_data
        gc.collect()

        # build model
        input_shape = (num_chunks,) + train_generator.__get_features_shape__()[-2:]
        model = chunk_based_rnn_model(input_shape=input_shape, num_output_neurons=num_classes,
                                      neurons_on_layer=(256, 256), rnn_type='LSTM', bidirectional=False,
                                      need_regularization=True, dropout=True)
        """model=chunk_based_1d_cnn_attention_model(input_shape=input_shape, num_output_neurons=num_classes,
                                       filters_per_layer = (128, 128, 128, 256,256),
                                       filter_sizes = (20, 15, 10, 6, 5),
                                       pooling_sizes = (8, 4,2,2,2),
                                       pooling_step = 1,
                                       need_regularization= True,
                                       dropout= True,
                                       dropout_rate=0.3
                                       )"""
        model.summary()
        #model.load_weights('C:\\Users\\Denis\\PycharmProjects\\AudioBasedEmotionRecognition\\src\\results\\MFCC_subwindow_3.000000_0.100000\\model_weights.h5', by_name=True)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(0.0005, decay=1e-6), loss='categorical_crossentropy',
                      metrics=[tf.keras.metrics.Recall()])
        # compute class weights
        train_labels = pd.read_csv(path_to_train_labels)
        class_weights = class_weight.compute_class_weight('balanced',
                                                          np.unique(train_labels['label'].values),
                                                          train_labels['label'].values)
        class_weights = {i: class_weights[i] for i in range(num_classes)}
        # train model
        hist=model.fit(train_generator, epochs=50,
                       callbacks=[validation_callback(devel_generator)], class_weight=class_weights)
        # collect the best reached score
        macro_recall=custom_recall_validation_with_generator(devel_generator, model)
        results.append((feature_type+'_macro_recall', macro_recall))
        print('feature_type:%s, max_val_MACRO_recall:%f' % (feature_type, macro_recall))
        # save predictions
        predictions=devel_generator.get_dict_predictions(model)
        df_for_saving_predictions=pd.DataFrame(data=np.zeros((len(predictions), num_classes + 1)))
        df_for_saving_predictions.iloc[:,0]=np.array(list(predictions.keys()))
        df_for_saving_predictions.iloc[:,1:]=np.array(list(predictions.values())).reshape((len(predictions), num_classes))
        df_for_saving_predictions.columns=['filename']+['class_prob_%i'%i for i in range(num_classes)]
        df_for_saving_predictions.to_csv(path_to_save_model+'predictions.csv', index=False)
        # save final results and weights
        pd.DataFrame(results, columns=['feature_type', 'val_recall']).to_csv(
            os.path.join(path_to_save_results, 'results_subwindow_%f_%f.csv' % (subwindow_size, subwindow_step)), index=False)
        model.save_weights(os.path.join(path_to_save_model, 'model_weights.h5'))

        # clear RAM
        del model
        #del hist
        del train_generator
        del devel_generator
        gc.collect()
        tf.keras.backend.clear_session()
    print(results)

def custom_recall_validation_with_generator(generator:ChunksGenerator_preprocessing, model:tf.keras.Model)->float:
    total_predictions=np.zeros((0,))
    total_ground_truth=np.zeros((0,))
    for x,y in generator:
        predictions=model.predict(x)
        predictions=predictions.argmax(axis=-1).reshape((-1,))
        total_predictions=np.append(total_predictions,predictions)
        total_ground_truth=np.append(total_ground_truth,y.argmax(axis=-1).reshape((-1,)))
    return recall_score(total_ground_truth, total_predictions,average='macro')

class validation_callback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator:ChunksGenerator_preprocessing):
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


if __name__ == '__main__':
    # params
    sequence_max_length = 12
    window_length = 0.5
    subwindow_lengths=(0.2,0.3)
    subwindow_steps=(0.1,0.2)
    for subwindow_length in subwindow_lengths:
        for subwindow_step in subwindow_steps:
            test_different_features(feature_types=('MFCC',),
                                    sequence_max_length=sequence_max_length, window_length=window_length,
                                    subwindow_size=subwindow_length,subwindow_step=subwindow_step)
