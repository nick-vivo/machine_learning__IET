import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

#Сбор данных

data = pd.read_csv("input.txt", sep=",")

x_train = []
y_train = []

for index, row in data.iterrows():
    if row['TT'] == "train":
        x_train.append([row['feature_0'], row['feature_1'], row['feature_2'], row['feature_3']])
        y_train.append(row['target'])

#Обучение модели

model = LogisticRegression(solver="lbfgs", multi_class='multinomial', C=5.0, random_state=0, max_iter=1500)

model.fit(x_train, y_train)

#Результаты тестов

x_test = []

for index, row in data.iterrows():
    if row['TT'] == "test":
        x_test.append([row['feature_0'], row['feature_1'], row['feature_2'], row['feature_3']])


z = model.predict_proba(x_test)

predicted_classes = np.argmax(z, axis=1)

with open("output.txt", "w") as f:
    for el in predicted_classes:
        f.write(str(el) + "\n")