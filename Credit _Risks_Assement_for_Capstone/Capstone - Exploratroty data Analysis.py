# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Liberaries

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, isnull
from pyspark.sql.functions import mean, col
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from pyspark.sql.types import StringType, BooleanType, DateType, IntegerType, DoubleType
from pyspark.sql.functions import when
from pyspark.sql.functions import expr
import numpy as np
import pyspark.pandas as ps

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Data

# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df1 = spark.read.table('hive_metastore.default.crs_transformed')
df1.printSchema()
display(df1)

# COMMAND ----------

row_count = df1.count()
column_count = len(df1.columns)

print("Number of rows:", row_count)
print("Number of columns:", column_count)

# COMMAND ----------

df1.columns
df1.printSchema()

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #Univariate

# COMMAND ----------

from pyspark.sql.functions import isnan, when, count, col, mean, stddev, min, max

# 1. Check data types
print("Data Types:")
df1.printSchema()



# COMMAND ----------

from pyspark.sql.functions import col

df1 = df1.withColumn("TNW_in_MEUR", col("TNW_in_MEUR").cast("integer"))
df1 = df1.withColumn("profits_perc_TNW", col("profits_perc_TNW").cast("integer"))
df1 = df1.withColumn("Positive_WC", col("Positive_WC").cast("integer"))
df1 = df1.withColumn("TNW_to_T-Exposure", col("TNW_to_T-Exposure").cast("integer"))
df1 = df1.withColumn("fleet_size", col("fleet_size").cast("integer"))
df1 = df1.withColumn("total_exposure", col("total_exposure").cast("integer"))
df1 = df1.withColumn("revenue", col("revenue").cast("integer"))
df1 = df1.withColumn("EBIT", col("EBIT").cast("integer"))
df1 = df1.withColumn("depreciation", col("depreciation").cast("integer"))
df1 = df1.withColumn("net_profit", col("net_profit").cast("integer"))
df1 = df1.withColumn("fixed_assets", col("fixed_assets").cast("integer"))
df1 = df1.withColumn("intangible_assets", col("intangible_assets").cast("integer"))
df1 = df1.withColumn("current_assets", col("current_assets").cast("integer"))
df1 = df1.withColumn("tangible_net_worth", col("tangible_net_worth").cast("integer"))
df1 = df1.withColumn("long_term_liab", col("long_term_liab").cast("integer"))
df1 = df1.withColumn("long_term_credit", col("long_term_credit").cast("integer"))
df1 = df1.withColumn("short_term_liab", col("short_term_liab").cast("integer"))
df1 = df1.withColumn("CR_Rating_score", col("CR_Rating_score").cast("integer"))
df1 = df1.withColumn("pmt_discipline_score", col("pmt_discipline_score").cast("integer"))
df1 = df1.withColumn("debt_equity_ratio", col("debt_equity_ratio").cast("integer"))

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA Univariabe Analysis

# COMMAND ----------

df1.columns

# COMMAND ----------

# 3. Basic statistics
numeric_cols = [item[0] for item in df1.dtypes if item[1] in ['integer', 'double']]

print("\nBasic Statistics:")
for col_name in numeric_cols:
    desc = df1.select(col_name).describe().toPandas()
    print(f"\n{col_name}:")
    print(desc)

# COMMAND ----------

import seaborn as sns

# Get the target variable distribution
target_distribution = df1.groupBy('target_variable').count().toPandas()

# Calculate percentage for each category
total_count = target_distribution['count'].sum()
target_distribution['percentage'] = (target_distribution['count'] / total_count) * 100

# Plot the distribution of the target variable
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=df1.toPandas(), x='target_variable')
plt.title('Distribution of Target')

# Annotate bars with percentages (adjusted position)
for i, count in enumerate(target_distribution['count']):
    percentage = target_distribution['percentage'][i]
    plt.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)

# Set x-axis label
plt.xlabel('Target')

# Show the plots
plt.tight_layout()
plt.show()


# COMMAND ----------

# Apply log transformation with np.log1p to handle columns with zero values
numerical_cols = ['intangible_assets', 'current_assets', 'tangible_net_worth', 'long_term_liab', 'long_term_credit', 'short_term_liab', 'CR_Rating_score', 'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio', 'current_ratio', 'return_on_assets',]

# COMMAND ----------

from pyspark.sql.functions import log1p, col

# Create a list of transformed columns
transformed_cols = [log1p(col(c)).alias(f"log1p_{c}") for c in numerical_cols]

# Select the transformed columns and convert to Pandas DataFrame
df_transformed = df1.select(*transformed_cols).toPandas()

# COMMAND ----------

# Plot distributions using histograms
df_transformed.hist(figsize=(20, 15))
plt.show()

# COMMAND ----------

# Compute correlation matrix
corr_matrix = df_transformed.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# COMMAND ----------

import numpy as np
from scipy import stats

# Detect outliers using Z-score
z_scores = df_transformed.apply(lambda x: np.abs(stats.zscore(x)), result_type='expand')
outliers = (z_scores > 3).sum()
print(outliers[outliers > 0])

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Convert the Pandas DataFrame to Spark DataFrame
df_spark = spark.createDataFrame(df_transformed)

for col in df_transformed.columns:
    # Distribution plot
    pdf = df_spark.select(col).toPandas()
    plt.figure(figsize=(3, 2))
    sns.histplot(pdf, bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

    # Box plot
    pdf = df_spark.select(col).toPandas()
    plt.figure(figsize=(3, 2))
    sns.boxplot(data=pdf)
    plt.title(f"Box Plot of {col}")
    plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot matrix
sns.pairplot(df_transformed, diag_kind='kde')
plt.show()

