# Databricks notebook source
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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import BinaryLogisticRegressionSummary

# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df = spark.read.table('hive_metastore.default.crs_transformed')
df.printSchema()
display(df)

# COMMAND ----------

from pyspark.sql.functions import when, count, col

# Array of column names to check  
cols_to_check = [
"YRS_In_BUS",
"TNW_in_MEUR",
"profits_perc_TNW",
"Positive_WC",
"TNW_to_T-Exposure",
"fleet_size",
"total_exposure",
"revenue",
"EBIT",
"depreciation",
"net_profit",
"fixed_assets",
"intangible_assets",
"current_assets",
"tangible_net_worth",
"long_term_liab",
"long_term_credit",
"short_term_liab",
"short_term_credit",
"off_balance_liab",
"CR_Rating_score",
"pmt_discipline_score",
"debt_equity_ratio",
"debt_asset_ratio",
"interest_cov_ratio",
"current_ratio",
"return_on_assets",
"debt_ser_cov_ratio",
"risk",
"risk_num"]

# Check each column for nulls
for col_name in cols_to_check:
  
  # Count nulls and non-nulls 
  aggregated = df.agg(
    count(when(col(col_name).isNull(), True)).alias("num_nulls"),
    count(when(col(col_name).isNotNull(), True)).alias("num_not_nulls")  
  ).toPandas()

  # Print results    
  print(f"{col_name}")
  print(aggregated) 
  print("")

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

# Select relevant columns for the logistic regression model
selected_columns = [
"YRS_In_BUS",
"TNW_in_MEUR",
"profits_perc_TNW",
"Positive_WC",
"TNW_to_T-Exposure",
"fleet_size",
"total_exposure",
"revenue",
"EBIT",
"depreciation",
"net_profit",
"fixed_assets",
"intangible_assets",
"current_assets",
"tangible_net_worth",
"long_term_liab",
"long_term_credit",
"short_term_liab",
"short_term_credit",
"off_balance_liab",
"CR_Rating_score",
"pmt_discipline_score",
"debt_equity_ratio",
"debt_asset_ratio",
"interest_cov_ratio",
"current_ratio",
"return_on_assets",
"debt_ser_cov_ratio"]

# Create a vector assembler to assemble features
assembler = VectorAssembler(inputCols=selected_columns[:-1], outputCol="features")

# Instantiate a StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

# Instantiate a logistic regression model
log_reg = LogisticRegression(featuresCol="scaled_features", labelCol="risk_num", family="multinomial")

# Create a pipeline with the assembler, scaler, and logistic regression stages
pipeline = Pipeline(stages=[assembler, scaler, log_reg])

# Split the data into training and testing sets
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=500)

# Fit the model on the training data
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model for accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="risk_num", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

# Print the accuracy
print(f"Accuracy: {accuracy}")

# Optionally, you can print precision, recall, and F1 score as well
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}")

# COMMAND ----------

# Access the logistic regression model
log_reg_model = model.stages[-1]

# Multiclass metrics
metrics = MulticlassMetrics(predictions.select("prediction", "risk_num").rdd)

# Access the coefficient matrix
coefficient_matrix = log_reg_model.coefficientMatrix
coefficient_matrix = log_reg_model.coefficientMatrix

# Access the intercept vector
intercepts = log_reg_model.interceptVector

# Print the coefficient matrix and intercepts
print("Coefficient Matrix:\n", coefficient_matrix)
print("Intercepts:", intercepts)

# COMMAND ----------

from pyspark.sql.functions import col

# Add probability prediction column
df_with_pd = model.transform(test_data).withColumn("probability", col("prediction"))

# Extract the probability 
df_with_pd.select("customer_name", "probability").show()


# COMMAND ----------


# Access the probability column in the predictions dataframe
probability_col = "probability"
default_probability = predictions.select("risk_num", probability_col)

# Print the default probability for each row
default_probability.show(truncate=False)

# Optionally, you can select the probability values for the "1.0" class (assuming binary classification)
# If you have multiple classes, adapt accordingly
probability_values = predictions.select("risk_num", probability_col).rdd.map(lambda row: (float(row.risk_num), float(row[probability_col][1])))
probability_values = spark.createDataFrame(probability_values, ["risk_num", "probability"])

# Print the probability values for the "1.0" class
probability_values.show(truncate=False)


# COMMAND ----------


# Access the probability column in the predictions dataframe
probability_col = "probability"

# Select relevant columns for constructing confusion matrix
prediction_and_labels = predictions.select("prediction", "risk_num").rdd.map(lambda row: (float(row.prediction), float(row.risk_num)))

# Create a MulticlassMetrics object
metrics = MulticlassMetrics(prediction_and_labels)

# Print the confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()
print("Confusion Matrix:")
print(confusion_matrix)


# COMMAND ----------

display(df.summary())
