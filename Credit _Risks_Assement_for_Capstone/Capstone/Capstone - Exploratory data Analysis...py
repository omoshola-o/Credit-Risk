# Databricks notebook source
# MAGIC %md
# MAGIC ## Import Liberaries

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# Convert Spark DataFrames to pandas DataFrames
df1 = df1.toPandas()

# COMMAND ----------

# MAGIC %md #Univariate

# COMMAND ----------

df_head = df1.head()
df_col = df1.columns
df_head, df_col

# COMMAND ----------

# Identifying variable types and checking for missing values
variable_types = df1.dtypes
missing_values = df1.isnull().sum()

# Summarize the findings
variable_types, missing_values

# COMMAND ----------

# Summary statistics for numerical variables
numerical_summary = df1.describe()
# Summarize the findings
numerical_summary

# COMMAND ----------

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count plot for 'Exited' column
sns.countplot(x='target_variable', data=df1, hue='target_variable', palette='Blues', ax=axes[1])
axes[1].set_title('Count Plot of target_variable')

# Pie chart for 'Exited' column
status_counts = df1['target_variable'].value_counts()
axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Blues'))
axes[0].set_title('Distribution of target_variable')



plt.tight_layout()
plt.show()

# COMMAND ----------

# Filter out the numerical columns
numerical_columns = ['TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC',
        'TNW_to_T-Exposure', 'fleet_size', 'total_exposure', 'revenue', 'EBIT',
        'depreciation', 'net_profit', 'fixed_assets', 'intangible_assets',
        'current_assets', 'tangible_net_worth', 'long_term_liab',
        'long_term_credit', 'short_term_liab', 'CR_Rating_score',
        'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio',
        'current_ratio', 'return_on_assets','target_variable']

# COMMAND ----------

# Filter out integer and float columns
numerical_columns = df1.select_dtypes(include=['int32', 'float64']).columns

# Computing the correlation matrix
correlation_matrix = df1[numerical_columns].corr()
correlation_matrix

# COMMAND ----------

# Correlation heatmap for numerical variables
plt.figure(figsize=(17, 13))
sns.heatmap(df1.select_dtypes(include=['int32', 'float64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Variables')
plt.show()

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# Filter out continuous columns
continuous_vars = [ 'CR_Rating_score','pmt_discipline_score']

# Define the number of rows and columns for subplots
num_rows = len(continuous_vars)
num_cols = 2  # Two plots for each column (box plot and KDE plot)

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))

# Flatten the axes array if there's only one row
if num_rows == 1:
    axes = axes.reshape(1, -1)

# Color for plots
color = 'skyblue'

# Iterate over each continuous variable and create box plots and KDE plots
for i, column in enumerate(continuous_vars):
    # Box plot
    sns.boxplot(x=df1[column], ax=axes[i, 0], color=color)
    axes[i, 0].set_title(f'Boxplot of {column}', fontsize=14)
    axes[i, 0].set_xlabel(column, fontsize=12)

    # KDE plot
    sns.kdeplot(data=df1[column], ax=axes[i, 1], color=color, fill=True)
    axes[i, 1].set_title(f'KDE Plot of {column}', fontsize=14)
    axes[i, 1].set_xlabel(column, fontsize=12)
    axes[i, 1].legend([column], loc='upper right')

# Adjust layout
plt.tight_layout()
plt.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col

# Selecting the two columns for analysis
columns = ['CR_Rating_score', 'pmt_discipline_score']


# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x='CR_Rating_score', y='pmt_discipline_score', data=df1, scatter_kws={'alpha':0.5})
plt.title('Relationship between CR Rating Score and Payment Discipline Score')
plt.xlabel('CR Rating Score')
plt.ylabel('Payment Discipline Score')
plt.grid(True)
plt.show()

