import pandas as pd
df=pd.read_csv("C:/Users/bhudi/Downloads/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df.head()
df[' Label'].value_counts()
df.columns
df2=df.drop('Flow Bytes/s', axis=1)
# Check for null values in each column
null_columns = df2.columns[df2.isnull().any()]

# Print the columns with null values
for col in null_columns:
    print(f"Column '{col}' has null values.")

len(df2)
from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# grid_search.fit(x_train, y_train)
# best_estimator = grid_search.best_estimator_
# print("Best Parameters:", grid_search.best_params_)
import numpy as np
df2.replace([np.inf, -np.inf], np.nan, inplace=True)
df3=df2.dropna()
df3.isnull().sum()
x=df3.drop(' Label', axis=1)
y=df3[' Label']
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=101)
import numpy as np

# Identify and drop rows with infinite values
x_train = x_train[~np.isinf(x_train).any(axis=1)]
y_train = y_train[~np.isinf(x_train).any(axis=1)]

# Check the shape of the data after dropping rows
print("Shape of X_train after dropping infinite values:",x_train.shape)
print("Shape of y_train after dropping infinite values:", y_train.shape)

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=100,random_state=101)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
ac=accuracy_score(y_test,y_pred)
print(ac)
cr=classification_report(y_test,y_pred)
cn=confusion_matrix(y_test,y_pred)
print(cn)
print(cr)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
sns.heatmap(cn,annot=True)
# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [None, 5, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
# grid_search.fit(x_train, y_train)

# best_rf_classifier = grid_search.best_estimator_

# y_pred2 = best_rf_classifier.predict(x_test)

# accuracy = accuracy_score(y_test, y_pred2)
# print("Accuracy:", accuracy)

# print("Best hyperparameters:", grid_search.best_params_)
