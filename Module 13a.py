#Predict if the loan application will get approved

#Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_graphviz

#Load dataset
url = "https://raw.githubusercontent.com/callxpert/datasets/master/Loan-applicant-details.csv"
names = ['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
dataset = pd.read_csv(url, names=names)
print(dataset.head(20))
#convert all our categorical variables into numeric by encoding the categories.
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
#Splitting the Data set
array = dataset.values
X = array[:,6:11]
Y=Y.astype('int')
Y = array[:,12]
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=7)
x_train=x_train.astype('int')
x_train
y_train=y_train.astype('int')
y_train
x_test=x_test.astype('int')
x_test
y_test=y_test.astype('int')
y_test
#Evaluating the model and training the Model
#Logistics Regression
l = LogisticRegression()
l.fit(x_train,y_train)
predictions = l.predict(x_test)
print(accuracy_score(y_test, predictions))
#Decison TRee
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))

#So, Regression algorithm works fine for our use case