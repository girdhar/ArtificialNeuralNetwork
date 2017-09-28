#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:20:25 2017

@author: girdharsinghbora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing datasets 
dataset = pd.read_csv('Churn_Modelling.csv')
#dataset.head(10)
dataset.shape
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:, 13:14].values
print(X.shape)

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
print(X.shape)
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:, 2])
print(X.shape)
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
print(X.shape)
print(X)
X = X[:, 1:]
print(X.shape)
print(X)

#Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print(X_train[1,:])
print(X_test[1,:])

#Importing Keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialasing the ANN
classifier = Sequential()

#Ading the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

#Adding second hidden layer

classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

#Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

#Compiling the ANN
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)


y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

#Make Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


new_pred = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred = (new_pred > 0.5)
print(new_pred)