# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 10:33:35 2019

@author: Rvi
"""

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importing the dataset
credit=pd.read_csv("F:\python/creditcard.csv")

#Visualization  of dataset
cor=credit.corr()
sns.heatmap(cor,linecolor='black',linewidths=0.1,cmap="coolwarm")

#calculation of number of fraud and valid transition 
Fraud = credit[credit['Class'] == 1]                               # 1 mean fraud transition
Valid = credit[credit['Class'] == 0]                              #0 mean valid transition
print(f"fraud  {type(Fraud)}  , Valid  {type(Valid)}")
outlier_fraction = len(Fraud)/float(len(Valid))                      #for calculating the outlier        
print("Outlier_fraction  :",outlier_fraction)
print('Fraud Cases: {}'.format(len(credit[credit['Class'] == 1])))        #print the total number of fraud transition
print('Valid Transactions: {}'.format(len(credit[credit['Class'] == 0])))  #print the total number of valid transition

# analysis the data
fig=plt.figure()
credit.hist(figsize=(21,20))
plt.plot()

X=credit.drop(columns=["Class"],axis=0)
y=credit.Class

# Splitting the dataset into Training Set and Test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.7,random_state=42)

'''from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X)
credit=sc.transform(X)
'''
#Fitting Linear Regression
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)

#Predicting The Test Set Result
from sklearn.metrics import classification_report,accuracy_score
y_pred=log_reg.predict(x_test)
print("Accuracy score using Logistic Regression :",accuracy_score(y_pred,y_test))
#Accuracy score using Logistic Regression : 0.9989265919293757


#Fitting Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gaussian=GaussianNB()
gaussian.fit(x_train,y_train)
pred=gaussian.predict(x_test)
print("Accuracy_score by Gaussian Naive",accuracy_score(y_test,pred))
#Accuracy_score by Gaussian Naive 0.994016000802548

#Fitting Multinomial Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
multinomial=GaussianNB()
multinomial.fit(x_train,y_train)
pred=multinomial.predict(x_test)
print("Accuracy score by Multinomial Naive",accuracy_score(y_test,pred))  
# Accuracy score by Multinomial Naive 0.994016000802548 






