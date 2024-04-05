# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decision tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KOUSALYA A.
RegisterNumber:  212222230068
*/
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform (data["Position"])
data.head()

x=data[["Position", "Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=2)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score (y_test,y_pred)
r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=x.columns, filled=True)
plt.show()

```

## Output:
![image](https://github.com/Kousalya22008930/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119389108/564bd8c5-d21f-4498-89c9-7fa51d77f982)
## MSE value:
![image](https://github.com/Kousalya22008930/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119389108/a368ab9d-7da1-4ed5-a0aa-f728207fb67a)
## R2 value:
![image](https://github.com/Kousalya22008930/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119389108/61abaa28-d527-40f3-af90-28fbc7b99c50)
## Predicted value:
![image](https://github.com/Kousalya22008930/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119389108/78c5c22b-6e26-4326-8d67-c3b839d5ef6b)
## Result tree:
![image](https://github.com/Kousalya22008930/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119389108/bb0d838e-78f8-4cb3-85f8-9c8915ea3098)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
