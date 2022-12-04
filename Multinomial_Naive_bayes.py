import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics._plot.confusion_matrix import confusion_matrix

df = pd.read_csv("/content/Beverage.csv")

"""print(df.head())
print(df.shape)
print(df.info())
print(df.describe())"""

df.quality = df.quality.astype('category').cat.codes

MNclsfr = MultinomialNB()
x = df.drop(columns = ["quality"])
y = df["quality"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 42)
MNclsfr.fit(xTrain, yTrain)

print("Train Score: ", MNclsfr.score(xTrain, yTrain))
print("Test score: ", MNclsfr.score(xTest, yTest))

yPred = MNclsfr.predict(xTest)
print("Prediction: ", yPred)
print(confusion_matrix(yTest, yPred))

print("Classification report\n", classification_report(yTest, yPred))

#when compearing 2 methods we alwyas check the macro avg F1 score