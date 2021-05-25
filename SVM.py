from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from prepared_data import X_train, X_test, Y_train, Y_test

#Support Vector Machine
from sklearn import svm
SVM = svm.SVC(C = 100, kernel = 'rbf', gamma = 0.01, probability = False).fit(X_train, Y_train)
Y_pred = SVM.predict(X_test)

#Evaluation
cm = confusion_matrix(Y_test, Y_pred)
print("\nSupport Vector Machine")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred)*100)
print('Precision: ' , precision_score(Y_test, Y_pred, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred, average='binary')*100))
