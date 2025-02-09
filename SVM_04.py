from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


bcancer = datasets.load_breast_cancer()

X = bcancer.data
Y = bcancer.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.20, random_state=0)

#Linear SVM
svmc = SVC(kernel='linear',random_state=0)
svmc.fit(Xtrain,Ytrain)
Ypred = svmc.predict(Xtest)
svmscore = accuracy_score(Ypred,Ytest)

print('Accuracy score of linear svm classifier is: ',100*svmscore,'%\n')


ksvmc = SVC(kernel='rbf',random_state=0)
ksvmc.fit(Xtrain,Ytrain)
Ypred = ksvmc.predict(Xtest)
svmscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of kernel svm classifier with RBF is: ',100*svmscore,'%\n')
