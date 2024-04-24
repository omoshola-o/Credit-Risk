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

df = df.drop('customer_name')
df.display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checking Data Imbalance

# COMMAND ----------


# Count the number of instances for each class
class_counts = df.groupBy("target_variable").count().orderBy("count", ascending=False).show()

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
predictions = lr_model.transform(train_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confussion Matrix

# COMMAND ----------

# Confusion Matrix
prediction_and_labels = predictions.select("prediction", predictions.target_variable.cast("double"))
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
# MAGIC ### Calculating the Precison and Racall

# COMMAND ----------

# Calculate precision and recall
true_positive = confusion_matrix[0][0]
false_positive = confusion_matrix[0][1]
false_negative = confusion_matrix[1][0]

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculating the Area Under ROC and Area Under PR

# COMMAND ----------

# Area Under ROC
evaluator = BinaryClassificationEvaluator().setLabelCol("target_variable").setRawPredictionCol("rawPrediction")
auc = evaluator.evaluate(predictions)
print("Area Under ROC:", auc)

# Area Under PR
evaluator = BinaryClassificationEvaluator().setLabelCol("target_variable").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
auprc = evaluator.evaluate(predictions)
print("Area Under PR:", auprc)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Making prediction with Test Data

# COMMAND ----------

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC # Checking Accuracy with other Models, to see which ones should be adopted and evaluated

# COMMAND ----------

# MAGIC %md
# MAGIC ### Support Vector Machines

# COMMAND ----------

# Create an SVM model
svm = LinearSVC(featuresCol="scaled_features", labelCol="target_variable")

# Train the SVM model
svm_model = svm.fit(train_data)

# Make predictions on test data
svm_predictions = svm_model.transform(test_data)

# Evaluate the SVM model
svm_evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
svm_accuracy = svm_evaluator.evaluate(svm_predictions)
print("SVM Accuracy:", svm_accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Bayes

# COMMAND ----------


# Create a Spark session
spark = SparkSession.builder \
    .appName("Naive Bayes") \
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

# Construct Naive Bayes model
nb = NaiveBayes(featuresCol="scaled_features", labelCol="target_variable", modelType="gaussian")

# Train the model
nb_model = nb.fit(train_data)

# Make predictions on test data
predictions = nb_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Naive Bayes Accuracy:", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree

# COMMAND ----------



# Create a Spark session
spark = SparkSession.builder \
    .appName("Decision Tree") \
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

# Construct Decision Tree model
dt = DecisionTreeClassifier(featuresCol="scaled_features", labelCol="target_variable")

# Train the model
dt_model = dt.fit(train_data)

# Make predictions on test data
predictions = dt_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Decision Tree Accuracy:", accuracy)
