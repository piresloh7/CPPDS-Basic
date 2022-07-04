import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
r=pd.read_csv("C:/Users/arun/Desktop/iris.csv")
r.head() #to check the first 10 rows of the data set
r.tail() #to check out last 10 row of the data set
r.describe() #to give a statistical summary about the dataset
r.sample(5) #pops up 5 random rows from the data set 
r.isnull().sum() #checks out how many null info are on the dataset
#plot
r.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
#Histogram
r.hist()
# Split-out validation dataset
array = r.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
#We have splited the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.
#Test harness
seed=7
scoring='accuracy'
#Building Models
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 kfold = model_selection.KFold(n_splits=10, random_state=seed)
 cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
 results.append(cv_results)
 names.append(name)
 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
 print(msg)
 
 
#OUTPUT 
# LR: 0.958333 (0.041667)
#LDA: 0.975000 (0.038188)
#KNN: 0.983333 (0.033333)
#CART: 0.983333 (0.033333)
#NB: 0.975000 (0.053359)
#SVM: 0.991667 (0.025000)
 
 # Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(metrics.accuracy_score(Y_validation, predictions))
print(metrics.confusion_matrix(Y_validation, predictions))
print(metrics.classification_report(Y_validation, predictions))