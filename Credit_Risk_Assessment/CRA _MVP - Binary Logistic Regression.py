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

# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.feature import ChiSqSelector
# from pyspark.ml.stat import ChiSquareTest

# # Step 1: Vectorize the features
# feature_cols = [col for col in df.columns if col != "target_variable"]
# assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
# vectorized_df = assembler.transform(df)

# # Step 2: Perform Chi-squared test to select features
# selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selected_features", labelCol="target_variable")
# selected_df = selector.fit(vectorized_df).transform(vectorized_df)

# # Step 3: Show the selected features
# df1 = selected_df.select("selected_features", "target_variable")
# df1.display()

# COMMAND ----------

from pyspark.sql.functions import round

df_describe =df.describe()
df_describe.display()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

# Create a SparkSession
spark = SparkSession.builder \
    .appName("StandardScalerExample") \
    .getOrCreate()


# Define the list of numeric column names (excluding the target variable)
numeric_cols = ["TNW_in_MEUR", "profits_perc_TNW", "Positive_WC", "TNW_to_T-Exposure",
                "fleet_size", "total_exposure", "revenue", "EBIT", "depreciation",
                "net_profit", "fixed_assets", "intangible_assets", "current_assets", 
                "tangible_net_worth", "long_term_liab", "long_term_credit", "short_term_liab",
                "CR_Rating_score", "pmt_discipline_score", "debt_equity_ratio", 
                "debt_asset_ratio", "current_ratio", "return_on_assets"]

# Vectorize the numeric columns
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
vectorized_df = assembler.transform(df)

# Instantiate the StandardScaler transformer
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Fit the scaler transformer to the data
scaler_model = scaler.fit(vectorized_df)

# Transform the data using the scaler model
scaled_df = scaler_model.transform(vectorized_df)

# Show the scaled features
scaled_df = scaled_df.select("scaled_features")
scaled_df.display()

vectorized_df.display()


# COMMAND ----------

# from pyspark.sql.functions import monotonically_increasing_id

# # Add a unique identifier column to both DataFrames
# scaled_df = scaled_df.withColumn("id", monotonically_increasing_id())
# df = df.withColumn("id", monotonically_increasing_id())

# # Join the two DataFrames on the id column
# joined_df = scaled_df.join(df.select("id", "target_variable"), "id", "inner").drop("id")

# # Select scaled_features and target_variable columns
# scaled_target_df = joined_df.select("scaled_features", "target_variable")

# # Show the DataFrame
# scaled_target_df.display()


# COMMAND ----------

scaled_df.printSchema()


# COMMAND ----------

# Split the data into training and testing sets
train_data, test_data = scaled_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Instantiate the logistic regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="target_variable")

# Fit the logistic regression model to the training data
lr_model = lr.fit(train_data)

# Make predictions on the testing data
predictions = lr_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="target_variable")
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)


# COMMAND ----------


# Evaluate the model using BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(labelCol="target_variable", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc_roc = evaluator.evaluate(predictions)

evaluator = BinaryClassificationEvaluator(labelCol="target_variable", rawPredictionCol="rawPrediction", metricName="areaUnderPR")
auc_pr = evaluator.evaluate(predictions)

print("AUC-ROC:", auc_roc)
print("AUC-PR:", auc_pr)

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

from pyspark.sql.functions import col

# Add probability prediction column
df_with_pd = model.transform(test_data).withColumn("probability", col("prediction"))

# Extract the probability 
df_with_pd.select("customer_name", "probability").show()


# COMMAND ----------

display(df.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC Accuracy: 0.896
# MAGIC The accuracy measures how often the model correctly predicts the positive or negative class. An accuracy of 0.896 means the model predicts the right class 89.6% of the time. This is pretty good but there may be some room for improvement.
# MAGIC
# MAGIC AUC-ROC: 0.896 
# MAGIC The AUC-ROC evaluates the model's ability to distinguish between positive and negative classes. A value of 0.896 indicates the model is fairly good at separating the two classes. Higher values closer to 1 are better, so there is some room to improve this model.
# MAGIC
# MAGIC AUC-PR: 0.741
# MAGIC The AUC-PR specifically looks at the balance between precision and recall. A score of 0.741 suggests a reasonable trade-off, but precision and recall could likely be improved.
# MAGIC
# MAGIC True Positive Rate/Recall: 0.511
# MAGIC The true positive rate or recall tells us that only 51.1% of the actual positive cases are being correctly classified by the model. We would prefer this to be higher, so the model is missing some real positive cases.
# MAGIC
# MAGIC False Positive Rate: 0.090 
# MAGIC The false positive rate is reasonably low at 9%, meaning only 9% of actual negatives are incorrectly classified as positive. We want this to be as low as possible.
# MAGIC
# MAGIC Precision: 0.673
# MAGIC Precision could be better at 67.3% - only about 2/3rd of predicted positives are actually positive. Reducing false positives can improve precision.
# MAGIC
# MAGIC In summary, the accuracy level is decent but the model can be improved to better detect positive cases, reduce false alarms, and improve the precision of its predictions. 
# MAGIC
# MAGIC
# MAGIC Advanced models would be adopted for some refinement.
