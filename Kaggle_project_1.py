import sklearn
from sklearn import tree
from sklearn import neighbors
from sklearn import svm

from google.colab import files
uploaded=files.upload()


import pandas as pd
import numpy as np

train = pd.read_csv('train.csv',header=None)
test = pd.read_csv('test.csv', header=None)

train.head()


trainLabels=pd.read_csv('trainLabels.csv', header=None)

trainLabels.head()



from sklearn import tree
from sklearn.model_selection import cross_val_score, train_test_split
train = pd.read_csv('train.csv',header=None)
test = pd.read_csv('test.csv', header=None)

x_train, x_test, y_train, y_test =train_test_split(train,trainLabels)

clf=tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x_train,y_train)


clf.score(x_train,y_train)


from sklearn.ensemble import RandomForestClassifier 
model=RandomForestClassifier(n_estimators=1000,criterion="entropy",n_jobs=-1,random_state=22,verbose=1)

model.fit(x_train,y_train)


model.score(x_test,y_test)



