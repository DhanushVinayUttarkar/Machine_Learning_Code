import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff

dataset = pd.read_csv("/content/CarResale.csv")

correlation = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=correlation.values,
    x=list(correlation.columns),
    y=list(correlation.index),
    annotation_text=correlation.round(2).values,
    showscale=True)
figure.show()

x = dataset.drop(['fuel', 'seller_type','transmission','owner', 'selling_price'], axis = 1) # Features
y = dataset['selling_price'] # Labels

Linreg = LinearRegression()

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

Linreg.fit(xTrain, yTrain)
print("prediction train: \n")
print(Linreg.score(xTrain, yTrain))
print("prediction test: \n")
print(Linreg.score(xTest, yTest))
ypred = Linreg.predict(xTest)
print(ypred)
