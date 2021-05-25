# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 21:10:28 2021

@author: Pariya
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle
from final_data import X_train, X_test, Y_train, Y_test

#Neural Networks (MLP)
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(random_state=1, max_iter=400).fit(X_train, Y_train)
Y_pred = MLP.predict(X_test)

#save model
filename = 'finalized_model.sav'
pickle.dump(MLP, open(filename, 'wb'))

#Evaluation
cm = confusion_matrix(Y_test, Y_pred)
print("\nMLP")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred)*100)
print('Precision: ' , precision_score(Y_test, Y_pred, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred, average='binary')*100))

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))