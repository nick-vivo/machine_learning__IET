import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

model = KNeighborsClassifier()

x_train_np = np.array(x_train)
y_train_np = np.array(y_train)

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
