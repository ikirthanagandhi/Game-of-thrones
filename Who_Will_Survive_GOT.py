#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:03:12 2019

@author: sarath
"""

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.naive_bayes import GaussianNB  
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


###############################
# Loading Data
###############################
got_df = pd.read_excel('GOT_character_predictions.xlsx')

# To check dimensions shape, ndim, dtype
got_df.shape
got_df.ndim

#only for numeric columns
got_df.describe() # we can draw some inferences out of this!!
got_df.info()

#For non-numeric columns
got_df['house'].value_counts()

"""
1) 62% are male characters out of 1946
2) 74.5% characters are Alive
"""

# We could drop the following columns as they wouldn't make any sense!

"""
dateOfBirth,mother, father, heir,spouse

"""

###############################
# CLeaning Data
###############################

#Removing all those columns which doesn't make sense/very low data/poor data quality as
# these vars wouldn't have any effect!!
df = got_df.drop(['S.No',
                  'dateOfBirth',
                  'mother',
                  'father',
                  'heir', 
                  'spouse', 
                  'isAliveMother', 'isAliveFather', 'isAliveHeir','isAliveSpouse'], 
                  axis = 1)
df.info()

#Checking distribution for Age as it could be important factor if a guy would be alive or not

df['age'].describe()  #minimum age is -298001 which is not possible! So removing negative values

df = df.drop(df[df.age < 0].index) #431 unique values

df.info()


##Combining factors
dummy = df
dummy['house'].count()

###############################
# EDA -> Explain about 5 or 6 variables
###############################

# Do EDA here - Some plots,
df['house'].value_counts() #Get some of the top counts and explain for other columns too!
df['isAlive'].value_counts()
df['title'].value_counts()
print(df['house'].value_counts().count()) #347 houses in total lol!!
# Example => Popularity box plot
# House => Counts/Histogram
# culture => counts/histogram
# Age => mean/median/mode
# combine isAlive, average(popularity) => Write something about it; hint: use groupby(check online)
# combine isAlive, male => write something about it; hint: same as above
# 


# Try any extra imputations here. 


###############################
# Checking correlation
###############################

# Check correlations
cor_mat = df.corr()
# Creating Heatmap
cmap = sns.diverging_palette(20, 10, as_cmap=True)
sns.heatmap(cor_mat, cmap = cmap, square = True) 

# use the above plot to explain how correlation works. I think there is no 
# strong correlation (> 0.5) is a good sign that we have chosen all independent 'X features

###############################
# Preparing data for the model
###############################

# Creating different columns for quick processing
ID_col = ['name']
target_col = ['isAlive']
cat_cols = ['title','male','culture','house']
num_cols= list(set(list(df.columns))-set(cat_cols)-set(ID_col)-set(target_col))

#Impute numerical missing values with mean
df[num_cols] = df[num_cols].fillna(round(df[num_cols].mean(),0),inplace=True)

#Imputing Categorical columns with forward fill
df[cat_cols] = df[cat_cols].ffill()
df = df.dropna() #dropping 2 rows as they have nulls due to forward fill!

#Checking if we have anymore null values
print(df.isnull().sum())

#2 rows affected by forward fill. so we put max value in it
#df[cat_cols] = df[cat_cols].fillna(df[cat_cols].value_counts().index[0]) 

#Tagging all factors into strings
df['culture'] = 'culture_' + df['culture'].astype(str)  
df['title'] = 'title_' + df['title'].astype(str)
df['house'] = 'house_' + df['house'].astype(str)

# Create Dummy variables and join
one_hot_culture = pd.get_dummies(df['culture'])  
df = df.join(one_hot_culture)
one_hot_title = pd.get_dummies(df['title'])  
df = df.join(one_hot_title)
one_hot_house = pd.get_dummies(df['house'])  
df = df.join(one_hot_house)

#Scaling numerical columns to avoid ordinality
#scaler = StandardScaler()
#df[num_cols] = scaler.fit_transform(df[num_cols]) 

#variable traps
df_final  = df.drop(['culture', 'title', 'house'], axis = 1) #dropped actual columns for 

#creating test & train
y = df_final[['isAlive', 'name']]
x = df_final.drop(['isAlive', 'name'], axis=1) #685

X_train, X_test, Y_train, Y_test = train_test_split(x,
                                                    y,
                                                    test_size = 0.25,
                                                    random_state = 508) 

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

###############################
# 4) Model Building 
###############################

#Creating feature set for model feeding
X_features = list(set(list(x.columns))-set(ID_col))
Y_features = list(set(list(y.columns))-set(ID_col))

#Logistic regression
logreg = LogisticRegression()
lr_model = logreg.fit(X_train[X_features], Y_train[Y_features])
lr_preds = logreg.predict(X_test[X_features])
lr_prediction = pd.DataFrame({'prediction' : lr_preds}) 

# Evaluate accuracy
print(accuracy_score(Y_test[Y_features], lr_preds))
lr_Result = pd.concat([Y_test.reset_index(drop=True), lr_prediction], axis=1)
lr_Result.head()

lr_Result.to_excel('result.xlsx', sheet_name='sheet1', index=False)


#KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn_model = knn.fit(X_train[X_features], Y_train[Y_features])
knn_preds = knn.predict(X_test[X_features])
knn_prediction = pd.DataFrame({'prediction' : knn_preds}) 

# Evaluate accuracy
print(accuracy_score(Y_test[Y_features], knn_preds))
knn_Result = pd.concat([Y_test.reset_index(drop=True), knn_prediction], axis=1)
knn_Result.head()


# Random Forest
random.seed(100)
rf = RandomForestClassifier(n_estimators=1000)
rf_model = rf.fit(X_train[X_features], Y_train[Y_features])
rf_preds = rf.predict(X_test[X_features])
rf_prediction = pd.DataFrame({'prediction' : rf_preds}) 
# Evaluate accuracy
print(accuracy_score(Y_test[Y_features], rf_preds))
rf_Result = pd.concat([Y_test.reset_index(drop=True), rf_prediction], axis=1)
rf_Result.head()


#Since random forest gives more accuracy in predicting a character Alive or Dead
# Results
print(confusion_matrix(Y_test[Y_features], rf_preds))
print(classification_report(Y_test[Y_features], rf_preds))

tn, fp, fn, tp = confusion_matrix(Y_test[Y_features], rf_preds).ravel()
(tn, fp, fn, tp)

#since random forest has more accuracy
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train[X_features].columns,
                                    columns=['importance']).sort_values('importance',    
                                                             ascending=False)
#Write this recommendation
result = feature_importances[feature_importances['importance'] >= 0.01]
result

########################################
#AUC
########################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# Logistic regression
lr_roc_auc = roc_auc_score(Y_test[Y_features], logreg.predict(X_test[X_features]))
lr_fpr, lr_tpr, lr_thresholds = roc_curve(Y_test[Y_features], logreg.predict_proba(X_test[X_features])[:,1])

# KNN(5)
knn_roc_auc = roc_auc_score(Y_test[Y_features], knn.predict(X_test[X_features]))
knn_fpr, knn_tpr, knn_thresholds = roc_curve(Y_test[Y_features], logreg.predict_proba(X_test[X_features])[:,1])

#RandomForest
rf_roc_auc = roc_auc_score(Y_test[Y_features], rf.predict(X_test[X_features]))
rf_fpr, rf_tpr, rf_thresholds = roc_curve(Y_test[Y_features], logreg.predict_proba(X_test[X_features])[:,1])


#Execute all together to get result
plt.figure(0).clf()
plt.figure()
plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
plt.plot(knn_fpr, knn_tpr, label='KNearest Neighbors (area = %0.2f)' % knn_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)
plt.legend(loc="lower right")
plt.show() #I couldnt get all 3 in one plot



# use AUC as evaluation metric for cross-validation 
from sklearn.cross_validation import cross_val_score 

#Reshaping the dataframe
c, r = y[Y_features].shape
y[Y_features] = y[Y_features].values.reshape(c,)

X = x[X_features].values
Y = y[Y_features].values

# CV score I Will work on this tmw!!
cross_val_score(logreg, X, Y, 
                cv=3, 
                scoring='roc_auc')


