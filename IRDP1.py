# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:32:08 2024

@author: ishma
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for step 3

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import ConfusionMatrixDisplay #for step 5
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

from sklearn.ensemble import StackingClassifier


#STEP 1

df = pd.read_csv("Project_1_Data.csv") #no need for file path since its in the downloads folder
print(df.info()) #check that line 12 actually worked

#STEP 2
plt.figure()
plt.scatter(df['Step'], df['X'], color='red', label = 'X')

plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Scatter Plot of X vs. Step')


plt.figure()

plt.scatter(df['Step'], df['Y'], color='green', label = 'Y')
plt.scatter(df['Step'], df['Z'], color='blue', label = 'Z')

plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Scatter Plot of Y and Z vs. Step') #Separate plots for X and Y,Z because of overlap making X impossible to see
plt.legend(loc='upper right')

#its clear that Z and X have a wide range of values at each step
avg_z = df.groupby('Step')['Z'].mean()

plt.figure()
plt.plot(avg_z.index,avg_z.values)
plt.xlabel('Step')
plt.ylabel('Average Z')
plt.title('Average Z at Each Step')

avg_x = df.groupby('Step')['X'].mean()

plt.figure()
plt.plot(avg_x.index,avg_x.values)
plt.xlabel('Step')
plt.ylabel('Average X')
plt.title('Average X at Each Step')


#STEP 3

#Data Splitting using simple method from lesson 3

X = df[['X','Y','Z']]
Y = df['Step']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

#Scaling... Update: Scaling ignored since it completely deletes any correlation between the features and the target variable
"""
my_scaler = StandardScaler()
my_scaler.fit(X_train)
scaled_data_train = my_scaler.transform(X_train)
scaled_data_train_df = pd.DataFrame(scaled_data_train,columns=X_train.columns)

scaled_data_test = my_scaler.transform(X_test)
scaled_data_test_df = pd.DataFrame(scaled_data_test,columns=X_test.columns)

X_train=scaled_data_train_df 
X_test=scaled_data_test_df
"""
#Correlation Matrix
plt.figure()
corr_matrix = (X_train).corr()
sns.heatmap(np.abs(corr_matrix))

corr1 = y_train.corr(X_train['X'])
print('The correlation between X and step is \n',corr1)
corr2 = y_train.corr(X_train['Y'])
print('The correlation between Y and step is \n',corr2)
corr3 = y_train.corr(X_train['Z'])
print('The correlation between Z and step is \n',corr3)

#STEP 4

my_model1 = LogisticRegression(random_state=42)
my_model1.fit(X_train,y_train)


param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga']
}

grid_search = GridSearchCV(my_model1, param_grid, cv=5, scoring = 'accuracy')
grid_search.fit(X_train, y_train)
best_params1 = grid_search.best_params_
print("Best Hyperparameters:", best_params1)
best_model1 = grid_search.best_estimator_
y_pred_train1 = best_model1.predict(X_train)
print("\n")

my_model2 = RandomForestClassifier(random_state=42)
my_model2.fit(X_train, y_train)


param_grid2 = {
    'n_estimators': [10, 30, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    }

grid_search2 = GridSearchCV(my_model2, param_grid2, cv=5, scoring = 'accuracy',n_jobs=1)
grid_search2.fit(X_train, y_train)
best_params2 = grid_search2.best_params_
print("Best Hyperparameters:", best_params2)
best_model2 = grid_search2.best_estimator_
y_pred_train2 = best_model2.predict(X_train)
print("\n")


my_model3 = GaussianNB()
my_model3.fit(X_train, y_train)


dist = { 
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
}

grid_search3 = RandomizedSearchCV(my_model3, dist,scoring = 'accuracy',random_state = 42)
grid_search3.fit(X_train, y_train)
best_params3 = grid_search3.best_params_
print("Best Hyperparameters:", best_params3)
best_model3 = grid_search3.best_estimator_
y_pred_train3 = best_model3.predict(X_train)

#STEP 5

ypred1 = best_model1.predict(X_test)
acc1 = accuracy_score(y_test, ypred1)
prec1 = precision_score(y_test, ypred1, average = 'weighted')
f11 = f1_score(y_test, ypred1, average='weighted')
print("\n")
print("For the logistic Regression Model \nThe accuracy is", acc1, "\nThe precision is", prec1, "\nThe f1 score is", f11)

ypred2 = best_model2.predict(X_test)
acc2 = accuracy_score(y_test, ypred2)
prec2 = precision_score(y_test, ypred2, average = 'weighted')
f12 = f1_score(y_test, ypred2, average='weighted')
print("\n")
print("For the Random Forest Classifier Model \nThe accuracy is", acc2, "\nThe precision is", prec2, "\nThe f1 score is", f12)

ypred3 = best_model3.predict(X_test)
acc3 = accuracy_score(y_test, ypred3)
prec3 = precision_score(y_test, ypred3, average = 'weighted')
f13 = f1_score(y_test, ypred3, average='weighted')
print("\n")
print("For the Gaussian Naive Bayes Model \nThe accuracy is", acc3, "\nThe precision is", prec3, "\nThe f1 score is", f13)

confusionmatrix1 = confusion_matrix(y_test,ypred1)
confusionmatrix2 = confusion_matrix(y_test,ypred2)
confusionmatrix3 = confusion_matrix(y_test,ypred3)

disp = ConfusionMatrixDisplay(confusionmatrix1)
disp.plot()
disp = ConfusionMatrixDisplay(confusionmatrix2)
disp.plot()
disp = ConfusionMatrixDisplay(confusionmatrix3)
disp.plot()

#STEP 6

estimators = [('lr', best_model1), ('rf', best_model2)]
clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(random_state=42),cv=5)
clf.fit(X_train,y_train)
ypred4 = clf.predict(X_test)

acc4 = accuracy_score(y_test, ypred4)
prec4 = precision_score(y_test, ypred4, average = 'weighted')
f14 = f1_score(y_test, ypred4, average='weighted')

confusionmatrix4 = confusion_matrix(y_test,ypred4)
disp = ConfusionMatrixDisplay(confusionmatrix4)
disp.plot()

print("\n")
print("For the Stacked Model \nThe accuracy is", acc4, "\nThe precision is", prec4, "\nThe f1 score is", f14)

