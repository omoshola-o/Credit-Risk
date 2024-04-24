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

plt.figure(figsize=(17, 13))
numerical_columnsss = ['revenue', 'EBIT',
        'depreciation', 'net_profit', 'fixed_assets', 'intangible_assets',
        'current_assets', 'tangible_net_worth', 'long_term_liab',
        'long_term_credit', 'short_term_liab', 'CR_Rating_score',
        'pmt_discipline_score',]
sns.heatmap(df1[numerical_columnsss].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Variables without Financial Ratio')
plt.show()

# COMMAND ----------

# Correlation heatmap for numerical variables
plt.figure(figsize=(17, 13))
sns.heatmap(df1.select_dtypes(include=['int32', 'float64']).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Numerical Variables')
plt.show()

# COMMAND ----------

# Filter out the numerical columns
numerical_columnss = ['TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC',
        'TNW_to_T-Exposure', 'fleet_size', 'total_exposure', 'long_term_liab',
        'long_term_credit', 'short_term_liab', 'CR_Rating_score',
        'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio',
        'current_ratio', 'return_on_assets','target_variable']

# COMMAND ----------

# Compute the correlation matrix
correlation_matrix = df1[numerical_columnss].corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Columns with Financial Ratio')
plt.show()

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


# COMMAND ----------

# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# from pyspark.sql.functions import corr

# # Define function for bivariate analysis
# def bivariate_analysis(data, columns):
#     # Pairplot with regression lines
#     sns.set(style="whitegrid")
#     plt.figure(figsize=(12, 10))
#     sns.pairplot(df1[columns], kind='reg')
#     plt.suptitle("Pairplot with Regression Lines", y=1.02)
#     plt.show()
    
#     # Box plots or Violin plots for categorical variables
#     categorical_columns = ['target_variable']  # Update with categorical variables if any
#     for col in categorical_columns:
#         plt.figure(figsize=(10, 6))
#         sns.violinplot(x=col, y='profits_perc_TNW', data=df1)
#         plt.title(f'Violin Plot: profits_perc_TNW by {col}')
#         plt.xlabel(col)
#         plt.ylabel('profits_perc_TNW')
#         plt.show()
    
#     # Joint plots
#     sns.set(style="white")
#     g = sns.jointplot(x='profits_perc_TNW', y='TNW_in_MEUR', data=df1, kind="hex", color="k")
#     g.set_axis_labels("Profits Percentage to TNW", "Total Net Worth in MEUR", fontsize=10)
#     plt.suptitle("Joint Plot: Profits Percentage to TNW vs Total Net Worth", y=1.02)
#     plt.show()
    
#     # Facet grids
#     sns.set(style="ticks")
#     g = sns.FacetGrid(df1, col="target_variable", hue="target_variable")
#     g.map(plt.scatter, "profits_perc_TNW", "TNW_in_MEUR", alpha=.7)
#     g.add_legend()
#     plt.suptitle("Facet Grid: Profits Percentage to TNW vs Total Net Worth by Target Variable", y=1.02)
#     plt.show()

# # Select the columns for analysis
# columns = ['TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC', 'TNW_to_T-Exposure', 'fleet_size', 
#            'total_exposure', 'revenue', 'EBIT', 'depreciation', 'net_profit', 'fixed_assets', 
#            'intangible_assets', 'current_assets', 'tangible_net_worth', 'long_term_liab', 
#            'long_term_credit', 'short_term_liab', 'CR_Rating_score', 'pmt_discipline_score', 
#            'debt_equity_ratio', 'debt_asset_ratio', 'current_ratio', 'return_on_assets', 'target_variable']

# # Perform bivariate analysis
# bivariate_analysis(df1, columns)


# COMMAND ----------

# import seaborn as sns

# # Get the target variable distribution
# target_distribution = df1.groupBy('target_variable').count().toPandas()

# # Calculate percentage for each category
# total_count = target_distribution['count'].sum()
# target_distribution['percentage'] = (target_distribution['count'] / total_count) * 100

# # Plot the distribution of the target variable
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# sns.countplot(data=df1.toPandas(), x='target_variable')
# plt.title('Distribution of Target')

# # Annotate bars with percentages (adjusted position)
# for i, count in enumerate(target_distribution['count']):
#     percentage = target_distribution['percentage'][i]
#     plt.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)

# # Set x-axis label
# plt.xlabel('Target')

# # Show the plots
# plt.tight_layout()
# plt.show()


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

