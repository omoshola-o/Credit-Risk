# Databricks notebook source
# MAGIC %md
# MAGIC # Install packages

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum, isnull, when, expr
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from pyspark.sql.types import StringType, BooleanType, DateType, IntegerType, DoubleType
import numpy as np


# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import BinaryLogisticRegressionSummary
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, NaiveBayes, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


# COMMAND ----------

# MAGIC %md
# MAGIC # Load Data

# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df = spark.read.table('hive_metastore.default.crs_transformed')
df.printSchema()
display(df)

# COMMAND ----------



# COMMAND ----------

df = df.drop('customer_name')
df.display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Data Imbalance

# COMMAND ----------


# Count the number of instances for each class
class_counts = df.groupBy("target_variable").count().orderBy("count", ascending=False)
display(class_counts)

# COMMAND ----------

# Count the number of instances for each class
class_counts = df.groupBy("target_variable").count().orderBy("count", ascending=False).toPandas()

# Calculate the ratio
minority_count = class_counts["count"].values[1]  # Assuming the second row is the minority class
majority_count = class_counts["count"].values[0]  # Assuming the first row is the majority class
ratio = minority_count / majority_count

print(f"Ratio of minority class to majority class: {ratio:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Logistic Regression

# COMMAND ----------


# Create a Spark session
spark = SparkSession.builder \
    .appName("Logistic Regression") \
    .getOrCreate()

# Convert numeric columns to DataFrame
numeric_cols = ["TNW_in_MEUR", "profits_perc_TNW", "Positive_WC", "TNW_to_T-Exposure",
               "fleet_size", "total_exposure", "revenue", "EBIT", "depreciation",
               "net_profit", "fixed_assets", "intangible_assets", "current_assets", 
               "tangible_net_worth", "long_term_liab", "long_term_credit", "short_term_liab",
               "CR_Rating_score", "pmt_discipline_score", "debt_equity_ratio", 
               "debt_asset_ratio", "current_ratio", "return_on_assets"]

df_num = df.select(numeric_cols)

# Vectorize the data
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
vectorized_df = assembler.transform(df)

# Normalize the data
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaled_df = scaler.fit(vectorized_df).transform(vectorized_df)

# Split the data into training and testing data
(train_data, test_data) = scaled_df.randomSplit([0.7, 0.3], seed=42)

# Construct logistic regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="target_variable")

# Train the model
lr_model = lr.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction with Train Data

# COMMAND ----------

# Make predictions on trainning data
predictions_train = lr_model.transform(train_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions_train)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confussion Matrix

# COMMAND ----------

# Confusion Matrix
prediction_and_labels = predictions_train.select("prediction", predictions_train.target_variable.cast("double"))
metrics = MulticlassMetrics(prediction_and_labels.rdd)

# Get the confusion matrix as a NumPy array
confusion_matrix = metrics.confusionMatrix().toArray()

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Metrics

# COMMAND ----------

# Calculate AUC (Area Under the Receiver Operating Characteristic Curve)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target_variable")
auc = evaluator.evaluate(predictions_train)
print(f"AUC: {auc:.2f}")

# Calculate Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions_train)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(predictions_train)
print(f"Weighted Precision: {precision:.2f}")

# Calculate Recall
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(predictions_train)
print(f"Weighted Recall: {recall:.2f}")

# Calculate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedFMeasure")
f1 = evaluator.evaluate(predictions_train)
print(f"Weighted F1 Score: {f1:.2f}")

# COMMAND ----------


import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from sklearn.metrics import roc_curve, auc

# Extracting probability of positive class and actual labels from predictions
raw_predictions = predictions_train.select('target_variable', 'probability') \
    .rdd.map(lambda row: (float(row['probability'][1]), float(row['target_variable'])))

# Compute ROC curve
roc = roc_curve(raw_predictions.map(lambda x: x[1]).collect(), raw_predictions.map(lambda x: x[0]).collect())
fpr = roc[0]
tpr = roc[1]

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction with Test Data

# COMMAND ----------

# Make predictions on test data
predictions_test = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions_test)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Support Vector Machines

# COMMAND ----------

# Create an SVM model
svm = LinearSVC(featuresCol="scaled_features", labelCol="target_variable")

# Train the SVM model
svm_model = svm.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction with Train Data

# COMMAND ----------

# Make predictions on train data
svm_predictions_train = svm_model.transform(train_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(svm_predictions_train)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix

# COMMAND ----------

# Confusion Matrix
prediction_and_labels = svm_predictions_train.select("prediction", svm_predictions_train.target_variable.cast("double"))
metrics = MulticlassMetrics(prediction_and_labels.rdd)

# Get the confusion matrix as a NumPy array
confusion_matrix = metrics.confusionMatrix().toArray()

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Metrics

# COMMAND ----------

# Calculate AUC (Area Under the Receiver Operating Characteristic Curve)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target_variable")
auc = evaluator.evaluate(svm_predictions_train)
print(f"AUC: {auc:.2f}")

# Calculate Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(svm_predictions_train)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(svm_predictions_train)
print(f"Weighted Precision: {precision:.2f}")

# Calculate Recall
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(svm_predictions_train)
print(f"Weighted Recall: {recall:.2f}")

# Calculate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedFMeasure")
f1 = evaluator.evaluate(svm_predictions_train)
print(f"Weighted F1 Score: {f1:.2f}")

# COMMAND ----------

# import matplotlib.pyplot as plt
# from pyspark.sql.functions import col
# from sklearn.metrics import roc_curve, auc

# # Extracting probability of positive class and actual labels from predictions
# raw_predictions = svm_model.select('target_variable', 'probability') \
#     .rdd.map(lambda row: (float(row['probability'][1]), float(row['target_variable'])))

# # Compute ROC curve
# roc = roc_curve(raw_predictions.map(lambda x: x[1]).collect(), raw_predictions.map(lambda x: x[0]).collect())
# fpr = roc[0]
# tpr = roc[1]

# # Compute AUC
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making predictions with Test Data

# COMMAND ----------

from pyspark.sql import DataFrame

# Train the SVM model
svm_predictions_test = svm.fit(test_data).transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(svm_predictions_test)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Naive Bayes

# COMMAND ----------


# Construct Naive Bayes model
nb = NaiveBayes(featuresCol="scaled_features", labelCol="target_variable", modelType="gaussian")

# Train the model
nb_model = nb.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making predictions with Train Data

# COMMAND ----------

# Make predictions on test data
nb_predictions_train = nb_model.transform(train_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(nb_predictions_train)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confussion Matrix

# COMMAND ----------

# Confusion Matrix
prediction_and_labels = nb_predictions_train.select("prediction", nb_predictions_train.target_variable.cast("double"))
metrics = MulticlassMetrics(prediction_and_labels.rdd)

# Get the confusion matrix as a NumPy array
confusion_matrix = metrics.confusionMatrix().toArray()

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Matrix

# COMMAND ----------

# Calculate AUC (Area Under the Receiver Operating Characteristic Curve)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target_variable")
auc = evaluator.evaluate(nb_predictions_train)
print(f"AUC: {auc:.2f}")

# Calculate Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(nb_predictions_train)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(nb_predictions_train)
print(f"Weighted Precision: {precision:.2f}")

# Calculate Recall
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(nb_predictions_train)
print(f"Weighted Recall: {recall:.2f}")

# Calculate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedFMeasure")
f1 = evaluator.evaluate(nb_predictions_train)
print(f"Weighted F1 Score: {f1:.2f}")

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from sklearn.metrics import roc_curve, auc

# Extracting probability of positive class and actual labels from predictions
raw_predictions = nb_predictions_train.select('target_variable', 'probability') \
    .rdd.map(lambda row: (float(row['probability'][1]), float(row['target_variable'])))

# Compute ROC curve
roc = roc_curve(raw_predictions.map(lambda x: x[1]).collect(), raw_predictions.map(lambda x: x[0]).collect())
fpr = roc[0]
tpr = roc[1]

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making predictions with Test Data

# COMMAND ----------

# Make predictions on test data
nb_predictions_test = nb_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(nb_predictions_test)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Random Forest

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Initialize Random Forest Classifier
rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="target_variable")

# Train the model
rf_model = rf.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction with Train Data

# COMMAND ----------

# Make predictions on training data
rf_predictions_train = rf_model.transform(train_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(rf_predictions_train)

print("Random Forest Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion Matrix

# COMMAND ----------

# Confusion Matrix
prediction_and_labels = rf_predictions_train.select("prediction", rf_predictions_train.target_variable.cast("double"))
metrics = MulticlassMetrics(prediction_and_labels.rdd)

# Get the confusion matrix as a NumPy array
confusion_matrix = metrics.confusionMatrix().toArray()

# Plot the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Metrics

# COMMAND ----------

# Calculate AUC (Area Under the Receiver Operating Characteristic Curve)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="target_variable")
auc = evaluator.evaluate(rf_predictions_train)
print(f"AUC: {auc:.2f}")

# Calculate Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(rf_predictions_train)
print(f"Accuracy: {accuracy:.2f}")

# Calculate Precision
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluator.evaluate(rf_predictions_train)
print(f"Weighted Precision: {precision:.2f}")

# Calculate Recall
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedRecall")
recall = evaluator.evaluate(rf_predictions_train)
print(f"Weighted Recall: {recall:.2f}")

# Calculate F1 Score
evaluator = MulticlassClassificationEvaluator(labelCol="target_variable", predictionCol="prediction", metricName="weightedFMeasure")
f1 = evaluator.evaluate(rf_predictions_train)
print(f"Weighted F1 Score: {f1:.2f}")

# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col
from sklearn.metrics import roc_curve, auc

# Extracting probability of positive class and actual labels from predictions
raw_predictions = rf_predictions_train.select('target_variable', 'probability') \
    .rdd.map(lambda row: (float(row['probability'][1]), float(row['target_variable'])))

# Compute ROC curve
roc = roc_curve(raw_predictions.map(lambda x: x[1]).collect(), raw_predictions.map(lambda x: x[0]).collect())
fpr = roc[0]
tpr = roc[1]

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction on Test Data

# COMMAND ----------

# Make predictions on test data
rf_predictions_test = rf_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(rf_predictions_test)

print("Decision Tree Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Computational Efficiency

# COMMAND ----------

# Import the required libraries
import time

# Start the timer
start_time = time.time()

# Make predictions on the testing data
predictions_test = lr_model.transform(test_data)

# Stop the timer
end_time = time.time()

# Calculate the computational time
computational_time = end_time - start_time

print(f"Computational time: {computational_time:.2f} seconds")

# COMMAND ----------

# Import the required libraries
import time

# Start the timer
start_time = time.time()

# Train the SVM model
svm_predictions_test = svm.fit(test_data).transform(test_data)

# Stop the timer
end_time = time.time()

# Calculate the computational time
computational_time = end_time - start_time

print(f"Computational time: {computational_time:.2f} seconds")

# COMMAND ----------

# Import the required libraries
import time

# Start the timer
start_time = time.time()

# Make predictions on test data
nb_predictions_test = nb_model.transform(test_data)

# Stop the timer
end_time = time.time()

# Calculate the computational time
computational_time = end_time - start_time

print(f"Computational time: {computational_time:.2f} seconds")

# COMMAND ----------

# Import the required libraries
import time

# Start the timer
start_time = time.time()

# Make predictions on test data
rf_predictions_test = rf_model.transform(test_data)

# Stop the timer
end_time = time.time()

# Calculate the computational time
computational_time = end_time - start_time

print(f"Computational time: {computational_time:.2f} seconds")
