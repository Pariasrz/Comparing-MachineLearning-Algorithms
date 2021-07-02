from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import pickle
from prepared_data import X_train, X_test, Y_train, Y_test

#Neural Networks (MLP)
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(random_state=0, max_iter=600).fit(X_train, Y_train)
Y_pred = MLP.predict(X_test)

#save model
filename = 'finalized_model.sav'
pickle.dump(MLP, open(filename, 'wb'))

#Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=MLP.classes_)
disp.plot(cmap="Blues")

TN = int(cm[0][0]) #true negatives
FP = int(cm[0][1]) #false positives

#Evaluation
print("\nMLP")
print('Confusion Matrix:' , cm)
print('Accuracy: ', accuracy_score(Y_test, Y_pred)*100)
print('Precision: ' , precision_score(Y_test, Y_pred)*100)
print('Recall: ', recall_score(Y_test, Y_pred)*100)
print('F-score: ' ,(f1_score(Y_test, Y_pred')*100))
print('Specificity: ', (float(TN/(TN+FP)))*100)
print('MCC: ', (matthews_corrcoef(Y_test, Y_pred))*100)

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
