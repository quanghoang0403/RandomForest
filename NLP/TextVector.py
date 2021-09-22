import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

class NLPModel:
    def __init__(self, data, col_label):
        self.data = data
        self.col_label = col_label
        self.countVector = CountVectorizer()

    def ConvertDataframe(self):
        return self.MatrixHandle(self.data)

    def ConvertRow(self, sentence):
        newData = self.data.append({self.col_label:sentence}, ignore_index = True)
        temple_matrix = self.MatrixHandle(newData)
        return temple_matrix.tail(1).iloc[0].array #get last row

    def MatrixHandle(self, dataframe):
        cdf = self.countVector.fit_transform(dataframe[self.col_label].values.astype('U'))
        return pd.DataFrame(cdf.toarray(), columns = self.countVector.get_feature_names())