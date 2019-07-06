import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
bankdata = pd.read_csv('C:/Users/Paolo Ferraiuoli/Desktop/sklearn_tut/bill_authentication.csv')  
#print number of rows and columns
print(bankdata.shape)  
#print(bankdata.describe())
#print(bankdata.columns)
print(bankdata.head())

#dividing the data into attributes (X) and labels (y)
X = bankdata.drop('Class', axis=1)# drop() drops the "Class" column,
#which is the lable column
y = bankdata['Class']

#divide data into training and test sets using the model_selection library
#from Scikit-Learn library (train_test_split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)

#Classification task using the support vector classifier class (SVC)
#in the Scikit-Learns's svm library
from sklearn.svm import SVC
#kernel =linear, simplest case to classify linarly separable data
svclassifier = SVC(kernel = 'linear')
#fit the model to train the algorithm on the training data
svclassifier.fit(X_train, y_train)

#Making prediction
y_pred = svclassifier.predict(X_test)

#Evaluate the algorithm using Confusion matrix, precision, recall and F1 measures
#Scikit-Learn's metrics library contains the classification_report and
#confusion_matrix methods
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


