# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:45:02 2020

@author: Pariya
"""

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from final_data import X_train, X_test, Y_train, Y_test


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion="entropy", max_depth = 10).fit(X_train, Y_train)
Y_pred_1 = DT.predict(X_test)

#Evaluation
cm = confusion_matrix(Y_test, Y_pred_1)
print("\nDecisionTree")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred_1)*100)
print('Precision: ' , precision_score(Y_test, Y_pred_1, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred_1, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred_1, average='binary')*100))


#randomForest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 100).fit(X_train, Y_train)
Y_pred_2 = RF.predict(X_test)


#Evaluation
cm = confusion_matrix(Y_test, Y_pred_2)
print("\nRandom Forest")
print('Confusion Matrix:' , cm)
#print(Y_pred.predict_proba(X_test[3:4].reshape(1,-1)))
print('Accuracy: ', accuracy_score(Y_test, Y_pred_2)*100)
print('Precision: ' , precision_score(Y_test, Y_pred_2, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred_2, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred_2, average='binary')*100))


