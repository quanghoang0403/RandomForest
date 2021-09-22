
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from RF import Extension
from NLP import TextVector

class RandomForestModel:

    def __init__(self, path):
        head_data = Extension.ReadCSV(path)
        self.nlpModel = TextVector.NLPModel(head_data, "Text")
        tail_data = self.nlpModel.ConvertDataframe()
        self.data = Extension.ConcatDataframe(head_data, tail_data)
        self.X = self.data[Extension.GetLabelTraining(self.data)]
        self.y = self.data['Sentiment'] 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=1, random_state=5)
        self.clf = RandomForestClassifier(n_estimators=100)

    def Fit(self):
        self.clf.fit(self.X_train,self.y_train)

    def GetAccuracy(self):
        y_pred=self.clf.predict(self.X_test)
        return print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
    
    def Predict(self, input):
        label = self.clf.predict([input])
        print(label)
        return label