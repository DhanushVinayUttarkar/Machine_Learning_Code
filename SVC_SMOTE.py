# SVM Classifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Importing dataset and examining it
dataset = pd.read_csv("/content/diabetes.csv")
# pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
# print(dataset.head(20))
# print(dataset.shape)
# print(dataset.info())
# print(dataset.describe())

# Dividing dataset into label and feature sets
X = dataset.drop('Outcome', axis=1)  # Features
Y = dataset['Outcome']  # Labels

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Implementing Support Vector Classifier
# Tuning the kernel and regularization parameter and implementing cross-validation using Grid Search
model = Pipeline([
    ('balancing', SMOTE(random_state=101)),
    ('classification', SVC(random_state=1))
])

scorer = make_scorer(f1_score, average='macro')

grid_param = {'classification__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'classification__C': [.001, .01, .1, 1, 10]}
gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring=scorer, cv=5)

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_  # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)

from imblearn.over_sampling import SMOTE  # RandomOverSampler
from collections import Counter

oversample = SMOTE(sampling_strategy='minority')

# fit and apply the transform
X_over, y_over = oversample.fit_resample(X_scaled, Y)
print(Counter(Y))
# X_over.shape
print(Counter(y_over))

model1 = SVC(random_state=1)
# grid_param1 = {'kernel': ['linear','poly','rbf','sigmoid'], 'C': [.001,.01,.1,1,10,100]}

grid_param1 = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [.001, .01, .1, 1, 10]}
gd_sr1 = GridSearchCV(estimator=model1, param_grid=grid_param1, scoring=scorer, cv=5)

gd_sr1.fit(X_over, y_over)
best_parameters = gd_sr1.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr1.best_score_  # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)

yPred = gd_sr1.predict(xTest)

print("Classification report\n", classification_report(yTest, yPred))
