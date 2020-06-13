# -*- coding: utf-8 -*-
#/usr/local/lib/python3.6

import rdflib
import string
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
dataset = pd.read_csv("features.csv")

#Apresenta informações do dataset 
#dataset.info()
#print(dataset.head())
#print(dataset.shape)
#print(dataset.describe())


#Separando as features e a class
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:,-1]
#print(X)
#print(Y)

validation_size = 0.10
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=None)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



#SVMmodel = SVC()
#SVMmodel.fit(X_train, Y_train)
#predictions = SVMmodel.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))


#KNNmodel = KNeighborsClassifier()
#KNNmodel.fit(X_train, Y_train)
#predictions = KNNmodel.predict(X_validation)
#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))




