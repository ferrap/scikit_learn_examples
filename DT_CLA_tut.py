import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
bankdata = pd.read_csv('C:/Users/Paolo Ferraiuoli/Desktop/bill_authentication.csv')  
#print number of rows and columns
print(bankdata.shape)  
#print(bankdata.describe())
#print(bankdata.columns)
print(bankdata.head())

#divide data into attribute(X) and label(y)
X = bankdata.drop('Class',axis =1)
y = bankdata['Class']

#divide data into training and tests sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

#select model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#make prediction
y_pred = classifier.predict(X_test)

#evaluate algorithm
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))

