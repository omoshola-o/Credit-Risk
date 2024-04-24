# Databricks notebook source
# MAGIC %md
# MAGIC # Install packages

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, isnull
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from pyspark.sql.types import StringType,BooleanType,DateType,IntegerType
from pyspark.sql.functions import when
from pyspark.sql.functions import expr

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import BinaryLogisticRegressionSummary
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

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

# Make predictions on test data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)

# COMMAND ----------


# True Positive Rate (Recall)
tp = predictions.filter("target_variable = 1 AND prediction = 1").count()
actual_positive = predictions.filter("target_variable = 1").count()
tpr = tp / actual_positive

# False Positive Rate
fp = predictions.filter("target_variable = 0 AND prediction = 1").count()
actual_negative = predictions.filter("target_variable = 0").count()
fpr = fp / actual_negative

# Precision
precision = tp / (tp + fp)

# Recall
recall = tp / actual_positive

# Print the results
print("True Positive Rate (Recall):", tpr)
print("False Positive Rate:", fpr)
print("Precision:", precision)
print("Recall:", recall)

# COMMAND ----------


# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="target_variable", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = evaluator.evaluate(predictions)

evaluator = BinaryClassificationEvaluator(labelCol="target_variable", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
auc_pr = evaluator.evaluate(predictions)

print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)
