from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,  accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from prepared_data import X_train, X_test, Y_train, Y_test


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0, criterion="entropy", min_samples_leaf=50).fit(X_train, Y_train)
Y_pred_1 = DT.predict(X_test)

#Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred_1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DT.classes_)
disp.plot(cmap="Blues")

TN = int(cm[0][0])
FP = int(cm[0][1])

#Evaluation
print("\nDecisionTree")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred_1)*100)
print('Precision: ' , precision_score(Y_test, Y_pred_1, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred_1, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred_1, average='binary')*100))
print('Specificity: ', (float(TN/(TN+FP)))*100)
print('MCC: ', (matthews_corrcoef(Y_test, Y_pred_1))*100)


#randomForest
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators= 100).fit(X_train, Y_train)
Y_pred_2 = RF.predict(X_test)


#Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred_2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DT.classes_)
disp.plot(cmap="Blues")

TN = int(cm[0][0]) #true negatives
FP = int(cm[0][1]) #false positives

#Evaluation
print("\nRandom Forest")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred_2)*100)
print('Precision: ' , precision_score(Y_test, Y_pred_2, average='binary')*100)
print('Recall: ', recall_score(Y_test, Y_pred_2, average='binary')*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred_2, average='binary')*100))
print('Specificity: ', (float(TN/(TN+FP)))*100)
print('MCC: ', (matthews_corrcoef(Y_test, Y_pred_1))*100)


