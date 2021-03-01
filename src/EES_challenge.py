from typing import Dict

import tensorflow as tf
import pandas as pd
import numpy as np

from src.utils.generators import AudioFixedChunksGenerator


def read_labels(path:str) -> Dict[str, np.ndarray]:
    labels=pd.read_csv(path)
    result_dict={}
    for index, row in labels.iterrows():
        result_dict[row['filename']]=np.array(row['label']).reshape((1,1))
    return result_dict





if __name__ == '__main__':
    path_to_data='D:\\Databases\\Compare_2021_ESS\\wav\\train\\'
    path_to_labels='D:\\Databases\\Compare_2021_ESS\\lab\\train.csv'
    labels=read_labels(path_to_labels)
    generator = AudioFixedChunksGenerator(sequence_max_length=12, window_length=0.5,
                                          load_mode='path',
                                          load_path=path_to_data,
                                          data_preprocessing_mode='LLD',
                                          labels=labels, labels_type='sequence_to_one', batch_size=4)
    a=1+2

    for x, y in generator:
        a=1+2


