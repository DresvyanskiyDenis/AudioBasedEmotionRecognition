
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score

if __name__ == '__main__':
    path_to_devel_labels = 'E:\\Databases\\Compare_2021_ESS\\lab\\devel.csv'
    path_to_devel_data = 'E:\\Databases\\Compare_2021_ESS\\wav\\dev\\'
    devel_labels=pd.read_csv(path_to_devel_labels)
    print(devel_labels.info())
    print(devel_labels["label"].values)
    recall=recall_score(devel_labels["label"].values, np.zeros((devel_labels.shape[0],)), average="micro")
    print(recall)