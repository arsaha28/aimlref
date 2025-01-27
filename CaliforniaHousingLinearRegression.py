import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt

BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:11]
y = BosData.iloc[:, 13] # MEDV: Median value of owner-occupied homes in $1000s

# Write code here
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=5)
reg = LinearRegression()
reg.fit(X_train,y_train)

y_train_predict = reg.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train,y_train_predict))
r2 = r2_score(y_train,y_train_predict)

print('Train RMSE =', rmse)
print('Train R2 score =', r2)
print("\n")

# Write code here
y_test_predict = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test,y_test_predict))
r2 = r2_score(y_test,y_test_predict)

print('Test RMSE =', rmse)
print('Test R2 score =', r2)

plt.figure(figsize=(12,6))
plt.scatter(y_test,y_test_predict,color='blue',alpha=0.6)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='red',linestyle='--')
plt.title('Account vs Predicted Values')
plt.xlabel('Actual MEDV')
plt.ylabel('predicted MEDV')
plt.grid()
plt.show()

#incomplete ,solve this for california housing