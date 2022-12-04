import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.figure_factory as ff

dataset = pd.read_csv("/content/CarResale.csv")
pd.set_option('display.max_columns', None)  # to make sure you can see all the columns in output window

# df = df.corr() #find the corelated values
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
figure.show()

x = dataset.drop(['fuel', 'seller_type', 'transmission', 'owner', 'selling_price'], axis=1)  # Features
y = dataset['selling_price']  # Labels

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=42)

feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(x)

sgdr = SGDRegressor(random_state=1, penalty=None)
grid_param = {'eta0': [.0001, .001, .01, .1, 1], 'max_iter': [10000, 20000, 30000, 40000]}

gd_sr = GridSearchCV(estimator=sgdr, param_grid=grid_param, scoring='r2', cv=5)

gd_sr.fit(X_scaled, y)

ypred = gd_sr.predict(xTest)
print(ypred)

results = pd.DataFrame.from_dict(gd_sr.cv_results_)
print("Cross-validation results:\n", results)

best_parameters = gd_sr.best_params_
print("Best parameters: ", best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print("Best result: ", best_result)

best_model = gd_sr.best_estimator_
print("Intercept: ", best_model.intercept_)

print(pd.DataFrame(zip(X.columns, best_model.coef_), columns=['Features','Coefficients']).sort_values(by=['Coefficients'],ascending=False))