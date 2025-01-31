
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
  return 1/(1 + np.exp(-x))

purchaseData = pd.read_csv('Purchase_Logistic.csv')

X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

print(X.size)
print(Y.size)
scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.25,random_state=0)
logr = LogisticRegression(random_state=0)
logr.fit(Xtrain,Ytrain)
Ypred = logr.predict(Xtest)

cmat = confusion_matrix(Ytest,Ypred)

plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.suptitle('Purchased Data')
plt.xlabel('Scaled Age')
plt.ylabel('Scaled Income')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()