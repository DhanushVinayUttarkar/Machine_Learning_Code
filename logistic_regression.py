import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from sklearn.metrics import classification_report
from sklearn.metrics._plot.confusion_matrix import confusion_matrix

df = pd.read_csv("/content/ChurnPrediction.csv")


# Converting Categorical features into Numerical features
def converter(column):
    if column == 'Yes':
        return 1
    else:
        return 0


df['PastEmployee'] = df['PastEmployee'].apply(converter)
df['OverTime'] = df['OverTime'].apply(converter)
# print(df.info())

# df = df.corr() #find the corelated values
corrs = df.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()

LogReg = LogisticRegression()
x = df.drop(
    columns=["PastEmployee", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus"])
y = df["PastEmployee"]

print(x.shape)
print(y.shape)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)
LogReg.fit(xTrain, yTrain)
# LogReg.score(xTest, yTest)
print(LogReg.score(xTrain, yTrain))
print(LogReg.score(xTest, yTest))

yPred = LogReg.predict(xTest)
print(yPred)
confusion_matrix(yTest, yPred)

print(classification_report(yTest, yPred))
