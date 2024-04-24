# Databricks notebook source
# MAGIC %pip install catboost
# MAGIC

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df = spark.read.table('hive_metastore.default.crs_transformed')
df.printSchema()
display(df)

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #Logistic Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Trainning 

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Drop unnecessary string columns
df_numerical = df.drop(columns=['customer_name'])

# Split features (X) and target variable (y)
X = df_numerical.drop(columns=['target_variable'])
y = df_numerical['target_variable']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define numerical columns
numerical_columns = X.columns.tolist()

# Preprocessing pipeline for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)
    ])

# Define logistic regression model with normalization
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Define hyperparameter grid
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
    'classifier__penalty': ['l2']  # Regularization type
}

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate model performance
best_model_lr = grid_search.best_estimator_
train_accuracy = best_model_lr.score(X_train, y_train)
print("Train accuracy:", train_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

# Generate predictions on the train set
y_pred_lr = best_model_lr.predict(X_train)

# Generate classification report
report = classification_report(y_train, y_pred_lr)

# Print classification report
print("Classification Report:")
print(report)


# COMMAND ----------

# MAGIC %md
# MAGIC ###  Confussion Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_roc_curve

# Generate predictions on the train set
y_pred_lr = best_model_lr.predict(X_train)

# Generate confusion matrix
cm = confusion_matrix(y_train, y_pred_lr)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]  # assuming binary classification
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation on Test Set

# COMMAND ----------

# Evaluate model performance on test set
best_model_test = grid_search.best_estimator_
test_accuracy = best_model_test.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC #Support Vector Machine

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Trainning

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Drop unnecessary string columns
df_numerical = df.drop(columns=['customer_name'])

# Split features (X) and target variable (y)
X = df_numerical.drop(columns=['target_variable'])
y = df_numerical['target_variable']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define numerical columns
numerical_columns = X.columns.tolist()

# Preprocessing pipeline for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)
    ])

# Define SVM model with normalization
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Define hyperparameter grid
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10],  # Regularization parameter
    'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid']  # Kernel type
}

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate model performance
best_model_svm = grid_search.best_estimator_
train_accuracy = best_model_svm.score(X_train, y_train)
print("Train accuracy:", train_accuracy)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

# Generate predictions on the train set
y_pred_svm = best_model_svm.predict(X_train)

# Generate classification report
report = classification_report(y_train, y_pred_svm)

# Print classification report
print("Classification Report:")
print(report)


# COMMAND ----------

# MAGIC %md
# MAGIC ###  Confussion Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_roc_curve

# Generate predictions on the train set
y_pred_svm = best_model_svm.predict(X_train)

# Generate confusion matrix
cm = confusion_matrix(y_train, y_pred_svm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]  # assuming binary classification
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation on Test Set

# COMMAND ----------

# Evaluate model performance
best_model_svm = grid_search.best_estimator_
test_accuracy = best_model_svm.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC #Naive Bayes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Trainning

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Drop unnecessary string columns
df_numerical = df.drop(columns=['customer_name'])

# Split features (X) and target variable (y)
X = df_numerical.drop(columns=['target_variable'])
y = df_numerical['target_variable']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define numerical columns
numerical_columns = X.columns.tolist()

# Preprocessing pipeline for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns)
    ])

# Define Gaussian Naive Bayes classifier model with normalization
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GaussianNB())
])

# Define hyperparameter grid
param_grid = {
    
}

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate model performance
best_model_nv = grid_search.best_estimator_
test_accuracy = best_model_nv.score(X_train, y_train)
print("Train accuracy:", train_accuracy)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

# Generate predictions on the train set
y_pred_nv = best_model_nv.predict(X_train)

# Generate classification report
report = classification_report(y_train, y_pred_nv)

# Print classification report
print("Classification Report:")
print(report)


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ###  Confussion Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_roc_curve

# Generate predictions on the train set
y_pred_nv = best_model_nv.predict(X_train)

# Generate confusion matrix
cm = confusion_matrix(y_train, y_pred_nv)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]  # assuming binary classification
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Evaluation on Test Set

# COMMAND ----------

# Evaluate model performance
best_model_nv = grid_search.best_estimator_
test_accuracy = best_model_nv.score(X_test, y_test)
print("Test accuracy:", test_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Nueral Networks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Trainning
# MAGIC

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# Drop unnecessary string columns
df_numerical = df.drop(columns=['customer_name'])

# Split features (X) and target variable (y)
X = df_numerical.drop(columns=['target_variable'])
y = df_numerical['target_variable']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define neural network classifier model with adjusted parameters to prevent overfitting
model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50), alpha=0.0001, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate model performance
train_accuracy = model.score(X_train, y_train)
print("Train accuracy:", train_accuracy)


# COMMAND ----------

# MAGIC %md
# MAGIC ###Classification Report

# COMMAND ----------

from sklearn.metrics import classification_report

# Generate predictions on the test set
y_pred_nn = model.predict(X_train)

# Generate classification report
report = classification_report(y_train, y_pred_nn)

# Print classification report
print("Classification Report:")
print(report)


# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ###  Confussion Matrix

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_roc_curve

# Generate predictions on the test set
y_pred_nn = model.predict(X_train)

# Generate confusion matrix
cm = confusion_matrix(y_train, y_pred_nn)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = [0, 1]  # assuming binary classification
plt.xticks(tick_marks, ['Negative', 'Positive'])
plt.yticks(tick_marks, ['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()


# COMMAND ----------

# MAGIC %md
# MAGIC ###Evaluation on Test Set

# COMMAND ----------

# Evaluate model performance
test_accuracy = model.score(X_test, y_test)
print("Test accuracy:", test_accuracy)
