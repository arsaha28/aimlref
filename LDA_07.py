from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

DiabetesData = pd.read_csv('Diabetes.csv')

X = DiabetesData.iloc[:, [0, 1]].values
Y = DiabetesData.iloc[:, 2].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=5)

plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.suptitle('Original Diabetes Data')
plt.xlabel('Scaled Glucose')
plt.ylabel('Scaled Blood Pressure')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

lda = LinearDiscriminantAnalysis()
lda.fit(Xtrain,Ytrain)
Y_pred = lda.predict(X)

ldascore = accuracy_score(lda.predict(Xtest),Ytest)
print('Accuracy score of LDA classifier is: ',100*ldascore,'%\n')

plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=Y_pred)
plt.suptitle('Predicted Diabetes Data')
plt.xlabel('Scaled Glucose')
plt.ylabel('Scaled Blood Pressure')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()