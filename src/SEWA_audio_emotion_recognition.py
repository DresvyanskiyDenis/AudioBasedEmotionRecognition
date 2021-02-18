from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import os

from src.utils.data_preprocessing_utils import load_wav_file, get_trained_minmax_scaler, transform_data_with_scaler
from src.utils.label_preprocessing_utils import load_gold_shifted_labels, split_labels_dataframe_according_filenames


def load_and_split_labels(path:str) -> Dict[str,pd.DataFrame]:
    labels = load_gold_shifted_labels(path_to_labels)
    splitted_labels=split_labels_dataframe_according_filenames(labels)
    return splitted_labels

def load_all_wav_files(path:str)-> Dict[str,Tuple[int,np.ndarray]]:
    loaded_wav_files={}
    filenames=os.listdir(path)
    for filename in filenames:
        sample_rate,wav_file=load_wav_file(os.path.join(path,filename))
        if len(wav_file.shape)<2: wav_file=wav_file[..., np.newaxis]
        loaded_wav_files[filename.split(".")[0]]=wav_file
    return loaded_wav_files

def split_data_on_train_dev_test(data:dict,dev_filenames:List[str],test_filenames:List[str])->\
        Tuple[Dict[str,np.ndarray],Dict[str,np.ndarray],Dict[str,np.ndarray]]:

    # separate dev data
    dev_data = {}
    for dev_filename in dev_filenames:
        dev_data[dev_filename] = data.pop(dev_filename)

    # separate test data
    test_data = {}
    for test_filename in test_filenames:
        test_data[test_filename] = data.pop(test_filename)

    return data, dev_data, test_data

def split_labels_on_train_dev_test(labels:dict, dev_filenames:List[str],test_filenames:List[str])->\
        Tuple[Dict[str,pd.DataFrame],Dict[str,pd.DataFrame],Dict[str,pd.DataFrame]]:
    # separate dev labels
    dev_labels = {}
    for dev_filename in dev_filenames:
        dev_labels[dev_filename] = labels.pop(dev_filename)

    # separate test data
    test_labels = {}
    for test_filename in test_filenames:
        test_labels[test_filename] = labels.pop(test_filename)

    return labels, dev_labels, test_labels

if __name__=="__main__":
    # params
    path_to_data=r"E:\Databases\SEWA\Original\audio"
    path_to_labels=r"E:\Databases\SEWA\SEW_labels_arousal_100Hz_gold_shifted.csv"
    test_filenames=["SEW1123","SEW1124","SEW2223","SEW2224"]
    dev_filenames=["SEW1119","SEW1120","SEW1121","SEW1122",
                          "SEW2219","SEW2220","SEW2221","SEW2222"]
    # load labels and data
    labels=load_and_split_labels(path_to_labels)
    data=load_all_wav_files(path_to_data)
    # preprocess data
        # separate dev, test data
    train_data, dev_data, test_data=split_data_on_train_dev_test(data, dev_filenames, test_filenames)
    train_lbs, dev_lbs, test_lbs=split_labels_on_train_dev_test(labels, dev_filenames, test_filenames)
        # normalize data
    concatenated_train_data=np.concatenate([x for x in train_data.values()])
    train_scaler=get_trained_minmax_scaler(data=concatenated_train_data)
    for key, value in train_data.items():
        train_data[key]=transform_data_with_scaler(value, train_scaler)

    for key, value in dev_data.items():
        dev_data[key]=transform_data_with_scaler(value, train_scaler)

    for key, value in test_data.items():
        test_data[key] = transform_data_with_scaler(value, train_scaler)





