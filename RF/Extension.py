import pandas as pd
import numpy as np

def ReadCSV(path):
    return pd.read_csv(path, sep=',', encoding='latin-1')

def ConcatDataframe(head, tail):
    return pd.concat([head, tail.reindex(head.index)], axis=1)

def GetLabelTraining(data):
    label_training = data.columns.values
    label_training = np.delete(label_training, 0) 
    label_training = np.delete(label_training, 0)
    label_training = np.delete(label_training, 0)
    label_training = np.delete(label_training, 0)
    label_training = np.delete(label_training, 1)
    label_training = np.delete(label_training, 2)
    # print("label string: ")
    # print(len(label_training))
    return label_training