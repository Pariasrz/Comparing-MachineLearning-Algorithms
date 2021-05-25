# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 21:14:32 2021

@author: Pariya
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from final_data import X_train, X_test, Y_train, Y_test


#K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 200).fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)

#Evaluation
cm = confusion_matrix(Y_test, Y_pred)
print("\nK Nearest Neighbors")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred)*100)
print('Precision: ' , precision_score(Y_test, Y_pred, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred, average='binary')*100))