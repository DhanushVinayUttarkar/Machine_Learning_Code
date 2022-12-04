import pandas as pd
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # RandomOverSampler
from collections import Counter
from sklearn.metrics import confusion_matrix

# Importing dataset and examining it
dataset = pd.read_csv("/content/Beverage.csv")
"""pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head(20))
print(dataset.shape)
print(dataset.info())
print(dataset.describe())"""

dataset.quality = dataset.quality.astype('category').cat.codes

# Dividing dataset into label and feature sets
x = dataset.drop('quality', axis=1)  # Features
y = dataset['quality']  # Labels
# print(type(X))
# print(type(Y))
# print(X.shape)
# print(Y.shape)

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(x)

xTrain, xTest, yTrain, yTest = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

oversample = SMOTE(random_state=42, sampling_strategy='minority')

# fit and apply the transform
X_over, y_over = oversample.fit_resample(xTrain, yTrain)
# X_over.shape
print(Counter(y_over))

dt = tree.DecisionTreeClassifier()

dt.fit(X_over, y_over)

print("overfit Score: ", dt.score(X_over, y_over))
print("Test Score: ", dt.score(xTest, yTest))

yPred = dt.predict(xTest)
print("Prediction: ", yPred)
print(confusion_matrix(yTest, yPred))

print("Classification report\n", classification_report(yTest, yPred))
