import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('student-mat.csv', sep=";")

data = data[['studytime', 'failures', 'absences',
             'G1', 'G2', 'G3']]

data.dropna()

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=1)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

pred = regressor.predict(x_test)

#acc = regressor.score(x_test, y_test)
# print(acc)
# 84% accuracy rate in the model

with open("model.pickle", "wb") as f:
    pickle.dump(regressor, f)
