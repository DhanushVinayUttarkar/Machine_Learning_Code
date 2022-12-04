import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics._plot.confusion_matrix import confusion_matrix

df = pd.read_csv("/content/Beverage.csv")

"""print(df.head())
print(df.shape)
print(df.info())
print(df.describe())"""

df.quality = df.quality.astype('category').cat.codes

Gclsfr = GaussianNB()
x = df.drop(columns = ["quality"])
y = df["quality"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.25, random_state = 42)
Gclsfr.fit(xTrain, yTrain)

print("Train Score: ", Gclsfr.score(xTrain, yTrain))
print("Test score: ", Gclsfr.score(xTest, yTest))

yPred = Gclsfr.predict(xTest)
print("Prediction: ", yPred)
print(confusion_matrix(yTest, yPred))

print("Classification report\n", classification_report(yTest, yPred))

#when compearing 2 methods we alwyas check the macro avg F1 score