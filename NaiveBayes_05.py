import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


purchaseData = pd.read_csv('Purchase_Logistic.csv')

X = purchaseData.iloc[:, [2, 3]].values
Y = purchaseData.iloc[:, 4].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=7)

cf = GaussianNB()
cf.fit(Xtrain,Ytrain)
Ypred = cf.predict(Xtest)

NBscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Naive Bayes classifier is: ',100*NBscore,'%\n')