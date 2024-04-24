# Databricks notebook source
# MAGIC %md
# MAGIC ### Install packages

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, isnull
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from pyspark.sql.types import StringType,BooleanType,DateType,IntegerType,DoubleType
from pyspark.sql.functions import when
from pyspark.sql.functions import expr
from pyspark.sql.functions import mean


# COMMAND ----------

# MAGIC %md
# MAGIC ### Laod data 

# COMMAND ----------

CONTRACT_TABLE = '[dbo].[Credit_data_all_markets]'

# Load Data
spark = SparkSession.builder \
    .master('local[1]') \
    .appName('CreditRiskAssesment') \
    .getOrCreate()

jdbc_url = dbutils.secrets.get(scope='ss_vg-adl-vfsbb', key='datasciencestudio-connectionstring-jdbc')

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .jdbc(url=jdbc_url, table=CONTRACT_TABLE)

display(df)
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preprocessing

# COMMAND ----------

# Drop unwanted columns
df1 = df.drop('VAT_number', 'legal_form', 'fundation_month', 'fundation_day', 'fundation_year', 'fleet_own_trucks','fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 'fleet_vfs_EC', 'fleet_other_EC', 'Offer_number_1', 'Offer_number_2', 'Offer_number_3','financial_performance_currency','type_of_request', 'flag_domestic_use','vfs_customer','vfs_known_since','quote_id', 'type_of_data')
display(df1)

# COMMAND ----------

# Dictionary mapping original column names -> new names  
renames = {
"customer_name": "customer_name",
"credit_rating": "CR_Rating",
"years_in_business": "YRS_In_BUS",
"TNW_in_MEUR": "TNW_in_MEUR",
"profits_perc_TNW": "profits_perc_TNW",
"positive_working_capital": "Positive_WC",
"TNW_to_total_exposure": "TNW_to_T-Exposure",
"financing_currency": "financing_currency",
"payment_discipline": "pmt_discipline",
"total_exposure": "total_exposure",
"current_period": "current_period(CP)",
"current_period_number_months": "CP_number_months",
"current_period_revenue": "CP_revenue",
"current_period_EBIT": "CP_EBIT",
"current_period_depreciation": "CP_depreciation",
"current_period_net_profit": "CP_net_profit",
"current_period_fixed_assets": "CP_fixed_assets",
"current_period_intangible_assets": "CP_intangible_assets",
"current_period_current_assets": "CP_current_assets",
"current_period_tangible_net_worth": "CP_tangible_net_worth",
"current_period_long_term_liabilities": "CP_long_term_liab",
"current_period_long_term_credit": "CP_long_term_credit",
"current_period_short_term_liabilities": "CP_short_term_liab",
"current_period_short_term_credit": "CP_short_term_credit",
"current_period_off_balance_liabilities": "CP_off_balance_liab",
"last_period": "last_period(LP)",
"last_period_number_months": "LP_number_months",
"last_period_revenue": "LP_revenue",
"last_period_EBIT": "LP_EBIT",
"last_period_depreciation": "LP_depreciation",
"last_period_net_profit": "LP_net_profit",
"last_period_fixed_assets": "LP_fixed_assets",
"last_period_intangible_assets": "LP_intangible_assets",
"last_period_current_assets": "LP_current_assets",
"last_period_tangible_net_worth": "LP_tangible_net_worth",
"last_period_long_term_liabilities": "LP_long_term_liab",
"last_period_long_term_credit": "LP_long_term_credit",
"last_period_short_term_liabilities": "LP_short_term_liab",
"last_period_short_term_credit": "LP_short_term_credit",
"last_period_off_balance_liabilities": "LP_off_balance_liab",
"period_before_last_period": "period_before_last_period(PBLP)",
"period_before_last_period_number_months": "PBLP_number_months",
"period_before_last_period_revenue": "PBLP_revenue",
"period_before_last_period_EBIT": "PBLP_EBIT",
"period_before_last_period_depreciation": "PBLP_depreciation",
"period_before_last_period_net_profit": "PBLP_net_profit",
"period_before_last_period_fixed_assets": "PBLP_fixed_assets",
"period_before_last_period_intangible_assets": "PBLP_intangible_assets",
"period_before_last_period_current_assets": "PBLP_current_assets",
"period_before_last_period_tangible_net_worth": "PBLP_tangible_net_worth",
"period_before_last_period_long_term_liabilities": "PBLP_long_term_liab",
"period_before_last_period_long_term_credit": "PBLP_long_term_credit",
"period_before_last_period_short_term_liabilities": "PBLP_short_term_liab",
"period_before_last_period_short_term_credit": "PBLP_short_term_credit",
"period_before_last_period_off_balance_liabilities": "PBLP_off_balance_liab",
"market": "market",
}

# Perform rename
for current_name, new_name in renames.items():  
    df1 = df1.withColumnRenamed(current_name, new_name)

# Show dataframe with new column names
df1.printSchema() 

# COMMAND ----------

# Some rows from the data has '0', 'null', 'nan' as the customer name, which did not represent any entity from the data. These items has no meaniningful variables accross the entire dataset. These rows will be dropped inorder to ascertain a meaningful sense of the data.

#Dropping list of unwanted records from the data
drop_list = ['0', 'null', 'nan']

df1 = df1.filter(~col("customer_name").isin(drop_list))
df1.display()

# COMMAND ----------

# check what data types each columns are
df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert numerical data to IntegerType format

# COMMAND ----------

# List of columns to convert to IntegerType
columns_to_integer = [
 'YRS_In_BUS',
 'TNW_in_MEUR',
 'profits_perc_TNW',
 'Positive_WC',
 'TNW_to_T-Exposure',
 'fleet_size',
 'total_exposure',
 'CP_number_months',
 'CP_revenue',
 'CP_EBIT',
 'CP_depreciation',
 'CP_net_profit',
 'CP_fixed_assets',
 'CP_intangible_assets',
 'CP_current_assets',
 'CP_tangible_net_worth',
 'CP_long_term_liab',
 'CP_long_term_credit',
 'CP_short_term_liab',
 'CP_short_term_credit',
 'CP_off_balance_liab',
 'LP_number_months',
 'LP_revenue',
 'LP_EBIT',
 'LP_depreciation',
 'LP_net_profit',
 'LP_fixed_assets',
 'LP_intangible_assets',
 'LP_current_assets',
 'LP_tangible_net_worth',
 'LP_long_term_liab',
 'LP_long_term_credit',
 'LP_short_term_liab',
 'LP_short_term_credit',
 'LP_off_balance_liab',
 'PBLP_number_months',
 'PBLP_revenue',
 'PBLP_EBIT',
 'PBLP_depreciation',
 'PBLP_net_profit',
 'PBLP_fixed_assets',
 'PBLP_intangible_assets',
 'PBLP_current_assets',
 'PBLP_tangible_net_worth',
 'PBLP_long_term_liab',
 'PBLP_long_term_credit',
 'PBLP_short_term_liab',
 'PBLP_short_term_credit',
 'PBLP_off_balance_liab',
]

# Convert specified columns to IntegerType
for column in columns_to_integer:
    df1 = df1.withColumn(column, col(column).cast(IntegerType()))

# Show the updated DataFrame
df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Date Data to DateType format

# COMMAND ----------

columns_to_date = [
    "Current_period(CP)",
    "last_period(LP)", 
    "period_before_last_period(PBLP)",
]

# Convert specified columns to DateType
for column in columns_to_date:
    df1 = df1.withColumn(column, col(column).cast(DateType()))

# Show the updated DataFrame
df1.printSchema()
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Group all data types to thier class for later use.

# COMMAND ----------

# Get column data types
schema = df1.schema

numerical_cols = [field.name for field in schema if isinstance(field.dataType, (IntegerType, DoubleType))]

categorical_cols = [field.name for field in schema if isinstance(field.dataType, StringType)] 

date_cols = [field.name for field in schema if isinstance(field.dataType, DateType)]

# Display categorized columns
categorized_columns = {
  "Numerical Columns": numerical_cols,
  "Categorical Columns": categorical_cols,
  "Date Columns": date_cols,
  }

for dtype, cols in categorized_columns.items():
  print(dtype)
  print(cols)
  print()

# COMMAND ----------

# from sklearn.impute import KNNImputer
# import numpy as np

# # 1. Identify numerical columns
# numerical_cols = df1.select_datatypes(include=[np.number]).columns.tolist()

# # 2. Filter numerical columns and pass to the imputer
# imputer = KNNImputer(n_neighbors=3)
# df1 = imputer.fit_transform(df[numerical_cols])

# COMMAND ----------

numerical_cols

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check all null values accoss each market segments

# COMMAND ----------

# Initialize an empty dictionary to store the null counts per column per market
null_counts_dict = {}

# Iterate over each column and calculate null counts
for column in df1.columns:
    null_counts = df1.groupBy("market").agg(
        F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
    )
    
    # Collect the null counts in the dictionary
    null_counts_dict[column] = null_counts

# Join the null count columns and sum to get the total per market 
aggregated = reduce(lambda df1, df2: df1.join(df2, on='market', how='inner'), null_counts_dict.values())

# Cast columns to integer
for col in aggregated.columns:
    if col != 'market':
        aggregated = aggregated.withColumn(col, aggregated[col].cast('int'))
        
# Sum all non-market columns to get the total null count per market
aggregated = aggregated.withColumn("total_nulls", 
                                   reduce(lambda x, y: x+y, [aggregated[c] for c in aggregated.columns if c != 'market']))

display(aggregated)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fill Null values with the mean of the data

# COMMAND ----------

# Calculate mean values for numerical columns
means = df1.select([mean(col).alias(col) for col in numerical_cols]).collect()[0].asDict()

# Fill null values with mean values
for col_name in numerical_cols:
    mean_value = means[col_name]
    df1 = df1.fillna(mean_value, subset=[col_name])

# Show the updated DataFrame
df1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Engineer Key Finanacial Ratios

# COMMAND ----------

from pyspark.sql.functions import col, expr

# Sum revenue across periods 
df1 = df1.withColumn("revenue", 
                   expr("CP_revenue + LP_revenue + PBLP_revenue"))

# Sum EBIT
df1 = df1.withColumn("EBIT",  
                   expr("CP_EBIT + LP_EBIT + PBLP_EBIT"))

# Sum depreciation
df1 = df1.withColumn("depreciation",  
                   expr("CP_depreciation + LP_depreciation + PBLP_depreciation"))

# Sum Net Profit
df1 =  df1.withColumn("net_profit",
                   expr("CP_net_profit + LP_net_profit + PBLP_depreciation"))

# Sum fixed asset
df1 = df1.withColumn("fixed_assets",
                   expr("CP_fixed_assets + LP_fixed_assets + PBLP_fixed_assets"))

# Sum Intangible assets
df1 = df1.withColumn("intangible_assets",
                   expr("CP_intangible_assets + LP_intangible_assets + PBLP_intangible_assets"))

# Sum current Assets
df1 = df1.withColumn("current_assets",
                   expr("CP_current_assets + LP_current_assets + PBLP_current_assets"))

# Sum Tangible Networth
df1 = df1.withColumn("tangible_net_worth",
                   expr("CP_tangible_net_worth + LP_tangible_net_worth + PBLP_tangible_net_worth"))

# Sum Long Term Liability
df1 = df1.withColumn("long_term_liab",
                   expr("CP_long_term_liab + LP_long_term_liab + PBLP_long_term_liab"))

# Sum Long Term Credit
df1 = df1.withColumn("long_term_credit",
                   expr("CP_long_term_credit + LP_long_term_credit + PBLP_long_term_credit"))

# Sum Short Term Liability
df1 = df1.withColumn("short_term_liab",
                   expr("CP_short_term_liab + LP_short_term_liab + PBLP_short_term_liab"))

# Sum Short Term Credit
df1 = df1.withColumn("short_term_credit",
                   expr("CP_short_term_credit + LP_short_term_credit + PBLP_short_term_credit"))

# Sum Off Balance Liability
df1 = df1.withColumn("off_balance_liab",
                   expr("CP_off_balance_liab + LP_off_balance_liab + PBLP_off_balance_liab"))

display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Map Credit Rating Column to Numerical Rating

# COMMAND ----------

columns = ["CR_Rating"]

# Define a mapping for credit ratings to numerical values
credit_rating_mapping = {"A": 100, "B": 75, "C": 50, "D": 25}

# Use the when and otherwise functions to create a new numerical column
df1 = df1.withColumn("CR_Rating_score", 
                   when(df1["CR_Rating"] == "A", credit_rating_mapping["A"])
                   .when(df1["CR_Rating"] == "B", credit_rating_mapping["B"])
                   .when(df1["CR_Rating"] == "C", credit_rating_mapping["C"])
                   .when(df1["CR_Rating"] == "D", credit_rating_mapping["D"])
                   .otherwise(None))

# Show the resulting DataFrame
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Map Payment Discipline Column to Numerical Rating

# COMMAND ----------

columns = ["pmt_discipline"]

# Define a mapping for payment discipline  to numerical values
payment_discipline_mapping = {"Excellent": 100, "Good": 75, "Fair": 50, "Bad": 25, "None": 25,}

# Use the when and otherwise functions to create a new target_variable column
df1 = df1.withColumn("pmt_discipline_score", 
                   when(df1["pmt_discipline"] == "Excellent", payment_discipline_mapping["Excellent"])
                   .when(df1["pmt_discipline"] == "Good", payment_discipline_mapping["Good"])
                   .when(df1["pmt_discipline"] == "Fair", payment_discipline_mapping["Fair"])
                   .otherwise(25))

# change the target variable to results from risk threshold.
display(df1)

# COMMAND ----------

# Debt to Equity Ratio
df1 = df1.withColumn(
    "debt_equity_ratio",
    expr(
      "(long_term_liab + short_term_liab) / (tangible_net_worth)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Debt to Assets Ratio
df1 = df1.withColumn(
    "debt_asset_ratio",
    expr(
      "(long_term_liab + short_term_liab) / (fixed_assets + intangible_assets + current_assets)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Interest Coverage Ratio
df1 = df1.withColumn(
    "interest_cov_ratio",
    expr(
      "(EBIT) / (short_term_credit)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Current Ratio
df1 = df1.withColumn(
    "current_ratio",
    expr(
      "(current_assets) / (short_term_liab)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Return on Assets (ROA)
df1 = df1.withColumn(
    "return_on_assets",
    expr(
      "(net_profit ) / (fixed_assets + intangible_assets + current_assets)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Debt Service Coverage Ratio (DSCR)
df1 = df1.withColumn(
    "debt_ser_cov_ratio",
    expr(
      "(net_profit + depreciation) / (short_term_credit + long_term_credit)"
    )
)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

# Drop unwanted columns
df1 = df1.drop ( 'CP_number_months', 'Current_period(CP)','last_period(LP)','period_before_last_period(PBLP)',
 'financing_currency', 'pmt_discipline', 'market','YRS_In_BUS',
 'CR_Rating',
 'CP_revenue',
 'CP_EBIT',
 'CP_depreciation',
 'CP_net_profit',
 'CP_fixed_assets',
 'CP_intangible_assets',
 'CP_current_assets',
 'CP_tangible_net_worth',
 'CP_long_term_liab',
 'CP_long_term_credit',
 'CP_short_term_liab',
 'CP_short_term_credit',
 'CP_off_balance_liab',
 'LP_number_months',
 'LP_revenue',
 'LP_EBIT',
 'LP_depreciation',
 'LP_net_profit',
 'LP_fixed_assets',
 'LP_intangible_assets',
 'LP_current_assets',
 'LP_tangible_net_worth',
 'LP_long_term_liab',
 'LP_long_term_credit',
 'LP_short_term_liab',
 'LP_short_term_credit',
 'LP_off_balance_liab',
 'PBLP_number_months',
 'PBLP_revenue',
 'PBLP_EBIT',
 'PBLP_depreciation',
 'PBLP_net_profit',
 'PBLP_fixed_assets',
 'PBLP_intangible_assets',
 'PBLP_current_assets',
 'PBLP_tangible_net_worth',
 'PBLP_long_term_liab',
 'PBLP_long_term_credit',
 'PBLP_short_term_liab',
 'PBLP_short_term_credit',
 'PBLP_off_balance_liab')
display(df1)

# COMMAND ----------

print(df1.columns)

# COMMAND ----------

# Get column data types
schema = df1.schema

numerical_cols2 = [field.name for field in schema if isinstance(field.dataType, (IntegerType, DoubleType))]

categorical_cols2 = [field.name for field in schema if isinstance(field.dataType, StringType)] 

date_cols2 = [field.name for field in schema if isinstance(field.dataType, DateType)]

# Display categorized columns
categorized_columns = {
  "Numerical Columns": numerical_cols2,
  "Categorical Columns": categorical_cols2,
  "Date Columns": date_cols2,
  }

for dtype, cols in categorized_columns.items():
  print(dtype)
  print(cols)
  print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Benchmarks and Thresholds for key financial metrics to use in assessing credit risk

# COMMAND ----------

# MAGIC %md
# MAGIC ### Liquidity Ratios

# COMMAND ----------

spark = SparkSession.builder.appName("CurrentRatio").getOrCreate()
# ----------------------------------------------------------------------------------------------------------------------------
# Current Ratio formula:
# Current Assets / Current Liabilities

# Check if the current ratio of the customer is at or above the 1.5x threshold. 

# Assumption:
# - If the customer has $2 million in current assets.
# - And $1 million in current liabilities.
# - Current ratio = Current Assets / Current liabilities 
#            = $2 million / $1 million  
#            = 2x

# This would meet the working capital adequacy benchmark.

# If the current ratio was instead only 1.1x ($2 million current assets / $1.8 million current liabilities), then
# flag the company for further assessment of their liquidity to meet short-term obligations.

# A minimum level of 1.5x could be reasonable threshold for adequate working capital.
# ----------------------------------------------------------------------------------------------------------------------------

# Calculate current ratio 
df1 = df1.withColumn("current_ratio", col("current_assets")/col("short_term_liab"))

# Filter companies below 1.5x ratio
min_ratio = 1.5

# Print results 
display(df1)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Debt/ Ratios

# COMMAND ----------

# spark = SparkSession.builder.appName("DebtEquityRatio").getOrCreate()

# # ----------------------------------------------------------------------------------------------------------------------------
# # Calculate Debt/Equity ratio using the formula:
# # Debt/Equity Ratio = Total Debt / Total Equity

# # Compare to the industry benchmark:
# #    - A debt/equity ratio above 2.0x indicates higher financial leverage and risk

# # Assumption: 
# # - If the customer has $5 million in total debt
# # - And $2 million in total shareholder's equity

# # Debt/Equity Ratio = Total Debt / Total Equity  
# #               = $5 million / $2 million
# #               = 2.5x

# # This is above the 2.0x benchmark, signaling high leverage and higher credit risk for this customer.
# # ----------------------------------------------------------------------------------------------------------------------------

# # Calculate debt/equity ratio
# df1 = df1.withColumn(
#     "debt_asset_ratio",
#     expr(
#       "(long_term_liab + short_term_liab) / (fixed_assets + intangible_assets + current_assets)"
#     )
# )

# # Benchmark debt/equity threshold  
# benchmark = 2.0


# # Print results 
# display(df1)

# COMMAND ----------

# # Calculate mean values for numerical columns
# means = df1.select([mean(col).alias(col) for col in numerical_cols]).collect()[0].asDict()

# # Fill null values with mean values
# for col_name in numerical_cols:
#     mean_value = means[col_name]
#     df1 = df1.fillna(mean_value, subset=[col_name])

# # Show the updated DataFrame
# df1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Target Variable Mapping

# COMMAND ----------


# Create a new column 'risk_category' based on risk thresholds
df1 = df1.withColumn(
    "target_variable",
    when(
        (col("CR_Rating_score") > 25) & (col("current_ratio") > 1.5),
        1
    ).otherwise(0)
)

# Where 1 is non_default, while 0 is default

# Show the resulting DataFrame with the new 'risk_category' column
display(df1)

# COMMAND ----------

df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Normalization

# COMMAND ----------

# MAGIC %md
# MAGIC # Calculate correlation coefficients

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Coeffients

# COMMAND ----------

from pyspark.sql.types import IntegerType, DoubleType 

# Identify integer and float columns 
int_cols = ['TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC', 'TNW_to_T-Exposure', 'fleet_size', 'total_exposure', 'revenue', 'EBIT', 'depreciation', 'net_profit', 'fixed_assets', 'intangible_assets', 'current_assets', 'tangible_net_worth', 'long_term_liab', 'long_term_credit', 'short_term_liab', 'short_term_credit', 'off_balance_liab', 'CR_Rating_score', 'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio', 'interest_cov_ratio', 'current_ratio', 'return_on_assets', 'debt_ser_cov_ratio']
float_cols = ['target_variable']

# Cast columns to appropriate type
cast_cols = []
for c in int_cols:
    cast_cols.append(F.col(c).cast(IntegerType()))

for c in float_cols:  
    cast_cols.append(F.col(c).cast(DoubleType()))

# Calculate correlation matrix  
correlation_matrix = df1.select(cast_cols).toPandas().corr()

print("Correlation Matrix:")
correlation_matrix.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use threshold from the correllation coefficient to carry out the features selection.

# COMMAND ----------

# Set a threshold for correlation coefficient
threshold = 0.8  # You can adjust this threshold based on your requirement

# Identify highly correlated features
highly_correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            highly_correlated_features.add(col1)
            highly_correlated_features.add(col2)

# Remove one of the features from each pair of highly correlated features
features_to_drop = list(highly_correlated_features)
df1_reduced = df1.drop(*features_to_drop)

# Display reduced DataFrame
print(features_to_drop)
print("Reduced DataFrame after feature selection:")
df1_reduced.display()

# COMMAND ----------

df1.printSchema()

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

# First, assemble the features into a vector
feature_columns = df1.columns
feature_columns.remove('customer_name')  # Remove non-numeric column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df1_assembled = assembler.transform(df1)

# Then, scale the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df1_assembled)
df_normalized = scaler_model.transform(df_assembled)

# Select only the necessary columns
df1_normalized = df1_normalized.select("customer_name", "scaled_features", "target_variable")

# Show the normalized DataFrame
df1_normalized.show(truncate=False)


# COMMAND ----------

from pyspark.sql.functions import col

# Iterate over each column and count the null values
null_counts = []
for col_name in df1.columns:
    null_count = df1.where(col(col_name).isNull()).count()
    null_counts.append((col_name, null_count))

# Convert the results to a DataFrame for better visualization
null_counts_df = spark.createDataFrame(null_counts, ["Column", "Null Count"])

# Show the DataFrame containing null counts
null_counts_df.show(25)


# COMMAND ----------

# Drop rows with null values in specific columns
columns_to_check = ['debt_equity_ratio', 'debt_asset_ratio', 'interest_cov_ratio', 'CR_Rating_score']
df1 = df1.dropna(subset=columns_to_check)

# Show the cleaned DataFrame with nulls dropped for specific columns
df1.display()


# COMMAND ----------

from pyspark.sql.functions import col

# Iterate over each column and count the null values
null_counts = []
for col_name in df1.columns:
    null_count = df1.where(col(col_name).isNull()).count()
    null_counts.append((col_name, null_count))

# Convert the results to a DataFrame for better visualization
null_counts_df = spark.createDataFrame(null_counts, ["Column", "Null Count"])

# Show the DataFrame containing null counts
null_counts_df.show(25)

# COMMAND ----------

# List of columns to select

cols_to_select = ['customer_name', 'TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC', 'TNW_to_T-Exposure', 'fleet_size', 'total_exposure', 'revenue', 'EBIT', 'depreciation', 'net_profit', 'fixed_assets', 'intangible_assets', 'current_assets', 'tangible_net_worth', 'long_term_liab', 'long_term_credit', 'short_term_liab', 'short_term_credit', 'off_balance_liab', 'CR_Rating_score', 'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio', 'interest_cov_ratio', 'current_ratio', 'return_on_assets', 'debt_ser_cov_ratio', 'target_variable']
   
df1 = df1.select(cols_to_select) 

# Print schema to verify columns
df1.printSchema()  

spark.sql('DROP TABLE IF EXISTS hive_metastore.default.crs_transformed')
df1.write.mode('overwrite').saveAsTable('hive_metastore.default.crs_transformed')

print("DataFrame written to Hive table successfully!")
