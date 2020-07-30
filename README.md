# Advertising-Logistcal-Regression-Analysis
Creating a machine learning model to predict whether a user will click an ad based off of the features of the user

#FIRST I'LL IMPORT THE LIBRARIES I NEED TO PERFORM ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#LOAD THE DATA TO A DATAFRAME & CHECK THE CONTENTS

ad_data = pd.read_csv('advertising.csv')
ad_data.describe()

#SOME EXPLORATORY DATA ANALYSIS.. GOING TO CHECK THE 'AGE' DISTRIBUTION TO VISUALIZE 
#THE COMMON AGE(S) WHERE AD WAS A SUCCESS

sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

#CHECK RELATIONSHIP BETWEEN THE AGE AND AREA OF INCOME & DAILY TIME ON SITE / DAILY INTERNET USAGE

sns.jointplot(x='Age',y='Area Income',data=ad_data)
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='red')

#PAIRPLOT WITH HUES BASED ON 'CLICKED ON AD' COLUMN
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

#NOW WE WILL LOOK AT THE LOGISTIC REGRESSION USING TRAIN_TEST_SPLIT (SPLIT THE DATA)

from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#TRAIN AND FIT A LOGISTIC REGRESSION FROM THE TRAINING DATA

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#NOW WE CAN PREDICT NEW VALUES

predictions = logmodel.predict(X_test)

#CREATE A CLASSIFICATION REPORT

from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
