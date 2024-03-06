# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph.
5.Predict the regression for the marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas .
```

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BHUVANESH S R
RegisterNumber:  212223240017
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_test,regressor.predict(X_test),color="red")
plt.title("Hours vs scores(Test Data Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)
rmse=np.sqrt(mse)
print("RMSE : ",rmse)
```

## Output:
## df.head()
![1](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/6ded435a-910a-4f7b-9956-19fca18fa635)

## df.tail()
![2](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/0b62a000-1029-4629-add8-2d5e68f0ee82)

## Values of X:
![3](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/f2faf9c5-7319-43dd-abf1-e900739fa167)

## Values of Y:
![4](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/baa26732-40d3-46f9-888a-eafe35d8112c)

## Values of Y prediction:
![5](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/6eab8d87-69f7-4ac6-a1c6-7d3f4abd8d20)

## Values of Y test:
![6](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/dcd9a9e4-c8a9-47b0-a40e-b5ba293489ed)

## Training set graph:
![7](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/11d63dd7-0e99-4804-b8b4-ec16d882ff95)

## Test set graph:
![8](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/9b138331-5501-48e8-bc43-baa054b8f0c7)

## Value of MSE,MAE & RMSE:
![9](https://github.com/Bhuvanesh-Suresh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742661/af20fcb5-35d1-4413-b562-6d59ce16703f)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
