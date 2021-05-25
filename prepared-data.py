from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv(r'data.csv')

datas = pd.DataFrame(data)


#find outliners
#print(datas.columns)
#plt.scatter('age_year','id', data = data, marker='o', color='lime')
#plt.scatter('height','id', data = data, marker='o', color='lime')
#plt.scatter('weight','id', data = data, marker='o', color='lime')
#plt.scatter('ap_hi','id', data = data, marker='o', color='lime')
#plt.scatter('ap_lo','id', data = data, marker='o', color='lime')

#remove outliners
data = data[data.age_year >= 35]
data = data[data.height <= 200]
data = data[data.height >= 120]
data = data[data.weight <= 170]
data = data[data.ap_hi <= 2000]
data = data[data.ap_lo <= 2000]

#remove unnecessary features
data.drop("id", axis=1, inplace = True)
data.drop("age_days", axis=1, inplace = True)


#normalization
scaler = MinMaxScaler()
data_norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

dataset = data_norm.to_numpy()[:70000]

X = dataset[:, :-1] # X
Y = dataset[:, -1] # Y

# split data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)
