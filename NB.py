from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score , matthews_corrcoef
from prepared_data import X_train, X_test, Y_train, Y_test

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
Y_pred = NB.fit(X_train, Y_train).predict(X_test)

#Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=NB.classes_)
disp.plot(cmap="Blues")

TN = int(cm[0][0]) #true negatives
FP = int(cm[0][1]) #false positives

#Evaluation
cm = confusion_matrix(Y_test, Y_pred)
print("\n Naive Bayes")
print('Confusion Matrix:', cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred)*100)
print('Precision: ' , precision_score(Y_test, Y_pred)*100)
print('Recall: ', recall_score(Y_test, Y_pred)*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred)*100))
print('Specificity: ', (float(TN/(TN+FP)))*100)
print('MCC: ', (matthews_corrcoef(Y_test, Y_pred))*100)
