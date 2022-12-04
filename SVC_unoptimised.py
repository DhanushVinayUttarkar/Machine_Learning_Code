import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics._plot.confusion_matrix import confusion_matrix

df = pd.read_csv("/content/Beverage.csv")

"""print(df.head())
print(df.shape)
print(df.info())
print(df.describe())"""

df.quality = df.quality.astype('category').cat.codes

svcclsft = make_pipeline(StandardScaler(), SVC(gamma='auto'))
x = df.drop(columns = ["quality"])
y = df["quality"]

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 42)
svcclsft.fit(xTrain, yTrain)

print("Train Score: ", svcclsft.score(xTrain, yTrain))
print("Test score: ", svcclsft.score(xTest, yTest))

yPred = svcclsft.predict(xTest)
print("Prediction: ", yPred)
print(confusion_matrix(yTest, yPred))

print("Classification report\n", classification_report(yTest, yPred))

#when compearing 2 methods we alwyas check the macro avg F1 score