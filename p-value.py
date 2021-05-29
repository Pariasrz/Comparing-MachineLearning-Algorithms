#from final_data import X_train, X_test, Y_train, Y_test
from scipy import stats
from NB import Y_pred as NB
from KNN import Y_pred as KNN
from MLP import Y_pred as MLP
from SVM import Y_pred as SVM
from DTRF import Y_pred_1 as DT
from DTRF import Y_pred_2 as RF


print("NB and KNN\n",stats.ttest_ind(NB,KNN))
print("\nNB and MLP\n",stats.ttest_ind(NB,MLP))
print("\nNB and SVM\n",stats.ttest_ind(NB,SVM))
print("\nNB and DT\n",stats.ttest_ind(NB,DT))
print("\nNB and RT\n",stats.ttest_ind(NB,RF))

print("\nMLP and KNN\n",stats.ttest_ind(MLP,KNN))
print("\nMLP and SVM\n",stats.ttest_ind(MLP,SVM))
print("\nMLP and DT\n",stats.ttest_ind(MLP,DT))
print("\nMLP and RF\n",stats.ttest_ind(MLP,RF))

print("\nKNN and SVM\n",stats.ttest_ind(KNN,SVM))
print("\nKNN and DT\n",stats.ttest_ind(KNN,DT))
print("\nKNN and RF\n",stats.ttest_ind(KNN,RF))

print("\nSVM and DT\n",stats.ttest_ind(SVM,DT))
print("\nSVM and RF\n",stats.ttest_ind(SVM,RF))

print("\nDT and RF\n",stats.ttest_ind(DT,RF))
