import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from RF import Extension
from NLP import TextVector

class RandomForestModel:

    def __init__(self, path):
        head_data = Extension.ReadCSV(path)
        self.path = path
        self.nlpModel = TextVector.NLPModel(head_data, "content_sentence")
        tail_data = self.nlpModel.ConvertDataframe()
        self.data = Extension.ConcatDataframe(head_data, tail_data)
        self.X = self.data[Extension.GetLabelTraining(self.data)]
        self.y = self.data['result_label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.8, random_state=5)
        self.clf = RandomForestClassifier(n_estimators=100)

    def Refresh(self):
        head_data = Extension.ReadCSV(self.path)
        self.nlpModel = TextVector.NLPModel(head_data, "content_sentence")
        tail_data = self.nlpModel.ConvertDataframe()
        self.data = Extension.ConcatDataframe(head_data, tail_data)
        self.X = self.data[Extension.GetLabelTraining(self.data)]
        self.y = self.data['result_label']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.8, random_state=5)
        self.clf = RandomForestClassifier(n_estimators=100)
        self.Fit()

    def Fit(self):
        self.clf.fit(self.X_train,self.y_train)

    def GetAccuracy(self):
        y_pred=self.clf.predict(self.X_test)
        return print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
    
    def Predict(self, input):
        label = self.clf.predict([input])
        return label

    def WriteBug(self, title, content):
        print(len(self.X_test))
        list_sentences = content.split(".")
        #read datacsv để lấy id của các bug cuối cùng => tạo id mới
        df  = pd.read_csv("./Dataset/final.csv", sep=',', encoding='latin-1')

        id_comment = 1
        pre = 0
        id_bug = 0
        if (id_bug == 0):
            id_bug = df.tail(1)["id_bug"].values[0] + 1
            
        id_sentences = 0
        if (id_sentences == 0):
            id_sentences = df.tail(1)["id_sentence"].values[0] + 1

        for index, item in enumerate(list_sentences):
            position = (index+1)/len(list_sentences)
            with open('./Dataset/final.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                convert_row = self.nlpModel.ConvertRow(position, item, pre, Extension.GetLabelTraining(self.data))
                result = self.Predict(convert_row)[0]
                writer.writerow([id_sentences, title, id_bug, id_comment, position, item, pre, result])
                pre = result
            id_sentences+=1
        df  = pd.read_csv("./Dataset/final.csv", sep=',', encoding='latin-1')    
        return df.tail(len(list_sentences))