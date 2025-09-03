import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

data=pd.read_csv('diabetes.csv')
data.head()
data.tail()
data.info()

data.isna().sum()
data.describe()

columns=['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in columns:
    mean=data[column].mean()
    data[column]=data[column].replace(0, mean)

###########
sns.heatmap(data.corr(),annot=True)

sns.displot(data['Age'])

sns.boxplot(y=data['Age'], x=data["Outcome"])


sns.boxplot(y=data['Pregnancies'], x=data["Outcome"])
sns.boxplot(y=data['Glucose'], x=data["Outcome"])

sns.boxplot(y=data['BloodPressure'], x=data["Outcome"])
sns.boxplot(y=data['SkinThickness'], x=data["Outcome"])

sns.boxplot(y=data['Insulin'], x=data["Outcome"])
sns.boxplot(y=data['BMI'], x=data["Outcome"])

###############

x=data.drop(['Outcome'], axis=1)
y=data["Outcome"]

#########3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.20, random_state=0)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
#classifier=KNeighborsClassifier()

classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_predict)

from sklearn.metrics import accuracy_score
print("KNN classifier accuracy score", accuracy_score(y_test, y_predict))

###############
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier()
classifier1.fit(x_train, y_train)

y_pred1 = classifier1.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred1)

from sklearn.metrics import accuracy_score
print("Random forest classifier accuracy score ", accuracy_score(y_test,y_pred1))

##############
import pickle
file = open('model.pkl', 'wb')
pickle.dump(classifier1, file)