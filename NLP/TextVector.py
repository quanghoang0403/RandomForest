import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class NLPModel:
    def __init__(self, data, col_label):
        self.data = data
        self.col_label = col_label
        self.countVector = CountVectorizer()

    def ConvertDataframe(self):
        return self.MatrixHandle(self.data)

    def ConvertRow(self, position, sentence, pre, label_training):
        # newData = self.data.append({self.col_label:sentence}, ignore_index = True)
        # temple_matrix = self.MatrixHandle(newData)
        # return np.concatenate([[position], temple_matrix.tail(1).iloc[0].array, [pre]]) #get last row
        sentence = sentence.lower()
        list_problem = ["not", "can't", "don't", "didn't" ,"?", "how", "why", "what","when","which", "try", "tried", "want","my"]
        list_solution = ["should", "will", "would", "need", "your","you","have to", "has to"]
        list_another = ["thank", "thanks", "interested" , "interest", "much", "nice"]
        for item in list_another:
            if (item in sentence):
                return 3
        for item in list_solution:
            if item in sentence:
                return 2
        for item in list_problem:
            if item in sentence:
                return 1
        if (pre==0):
            return 1
        final = [0]*len(label_training)
        final[0]=position
        final[-1]=pre
        for word in sentence.split(" "):
            for index, value in enumerate(label_training):
                if word == value:
                    final[index]=1
        return final

    def MatrixHandle(self, dataframe):
        cdf = self.countVector.fit_transform(dataframe[self.col_label].values.astype('U'))
        return pd.DataFrame(cdf.toarray(), columns = self.countVector.get_feature_names())