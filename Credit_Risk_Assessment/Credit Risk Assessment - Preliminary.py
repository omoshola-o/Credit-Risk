# Databricks notebook source
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, sum
from pyspark.sql.functions import col, isnull
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# COMMAND ----------

DEBUG = True

# COMMAND ----------

CONTRACT_TABLE = '[dbo].[Credit_data_all_markets]'

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

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.sql.functions import col

# List of columns to be cast to integer
columns_to_cast = ['years_in_business', 'TNW_in_MEUR', 'profits_perc_TNW', 'fleet_size', 'fleet_own_trucks',
                   'fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 'fleet_vfs_EC', 'fleet_other_EC',
                   'Offer_number_1', 'Offer_number_2', 'Offer_number_3', 'current_period_number_months',
                   'current_period_revenue', 'current_period_EBIT', 'current_period_depreciation',
                   'current_period_net_profit', 'current_period_fixed_assets', 'current_period_intangible_assets',
                   'current_period_current_assets', 'current_period_tangible_net_worth',
                   'current_period_long_term_liabilities', 'current_period_long_term_credit',
                   'current_period_short_term_liabilities', 'current_period_short_term_credit',
                   'current_period_off_balance_liabilities', 'last_period_number_months', 'last_period_revenue',
                   'last_period_EBIT', 'last_period_depreciation', 'last_period_net_profit', 'last_period_fixed_assets',
                   'last_period_intangible_assets', 'last_period_current_assets', 'last_period_tangible_net_worth',
                   'last_period_long_term_liabilities', 'last_period_long_term_credit', 'last_period_short_term_liabilities',
                   'last_period_short_term_credit', 'last_period_off_balance_liabilities',
                   'period_before_last_period_number_months', 'period_before_last_period_revenue',
                   'period_before_last_period_EBIT', 'period_before_last_period_depreciation',
                   'period_before_last_period_net_profit', 'period_before_last_period_fixed_assets',
                   'period_before_last_period_intangible_assets', 'period_before_last_period_current_assets',
                   'period_before_last_period_tangible_net_worth', 'period_before_last_period_long_term_liabilities',
                   'period_before_last_period_long_term_credit', 'period_before_last_period_short_term_liabilities',
                   'period_before_last_period_short_term_credit', 'period_before_last_period_off_balance_liabilities']


# Create a new DataFrame for each cast
for column in columns_to_cast:
    df_to_int = df.withColumn(column, col(column).cast("int"))

# Separate columns into integer
integer_columns = [col for col in columns_to_cast if df_to_int.select(col).dtypes[0][1] == 'int']

# Create DataFrames for integer
df_integer = df_to_int.select(integer_columns)

# Show the schemas of the new DataFrames
df_integer.printSchema()

# COMMAND ----------

# Get column data types
schema = df.schema

numerical_cols = [field.name for field in schema if isinstance(field.dataType, (IntegerType, FloatType, DoubleType))]

categorical_cols = [field.name for field in schema if isinstance(field.dataType, StringType)] 


# Display categorized columns
categorized_columns = {
  "Numerical Columns": numerical_cols,
  "Categorical Columns": categorical_cols,
  }

for dtype, cols in categorized_columns.items():
  print(dtype)
  print(cols)
  print()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing

# COMMAND ----------

# Some rows from the data has '0' as the customer name, which did not represent any entity from the data. These items has no meaniningful variables accross the entire dataset. These rows will be dropped inorder to ascertain a meaningful sense of the data
df_filtered = df.filter(df['customer_name']== "nan")
display(df_filtered)

# COMMAND ----------

df_new = df.filter(df['customer_name'] != "0")
display(df_new)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Numerical Variables to Integer

# COMMAND ----------

from pyspark.sql.functions import col

# List of columns to be cast to integer
columns_to_cast = ['years_in_business', 'TNW_in_MEUR', 'profits_perc_TNW', 'fleet_size', 'fleet_own_trucks',
                   'fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 'fleet_vfs_EC', 'fleet_other_EC',
                   'Offer_number_1', 'Offer_number_2', 'Offer_number_3', 'current_period_number_months',
                   'current_period_revenue', 'current_period_EBIT', 'current_period_depreciation',
                   'current_period_net_profit', 'current_period_fixed_assets', 'current_period_intangible_assets',
                   'current_period_current_assets', 'current_period_tangible_net_worth',
                   'current_period_long_term_liabilities', 'current_period_long_term_credit',
                   'current_period_short_term_liabilities', 'current_period_short_term_credit',
                   'current_period_off_balance_liabilities', 'last_period_number_months', 'last_period_revenue',
                   'last_period_EBIT', 'last_period_depreciation', 'last_period_net_profit', 'last_period_fixed_assets',
                   'last_period_intangible_assets', 'last_period_current_assets', 'last_period_tangible_net_worth',
                   'last_period_long_term_liabilities', 'last_period_long_term_credit', 'last_period_short_term_liabilities',
                   'last_period_short_term_credit', 'last_period_off_balance_liabilities',
                   'period_before_last_period_number_months', 'period_before_last_period_revenue',
                   'period_before_last_period_EBIT', 'period_before_last_period_depreciation',
                   'period_before_last_period_net_profit', 'period_before_last_period_fixed_assets',
                   'period_before_last_period_intangible_assets', 'period_before_last_period_current_assets',
                   'period_before_last_period_tangible_net_worth', 'period_before_last_period_long_term_liabilities',
                   'period_before_last_period_long_term_credit', 'period_before_last_period_short_term_liabilities',
                   'period_before_last_period_short_term_credit', 'period_before_last_period_off_balance_liabilities']


# Create a new DataFrame for each cast
df_to_int = df_new
for column in columns_to_cast:
    df_to_int = df_to_int.withColumn(column, col(column).cast("int"))

# Separate columns into integer
integer_columns = [col for col in columns_to_cast if df_to_int.select(col).dtypes[0][1] == 'int']

# Create DataFrames for integer
df_integer = df_to_int.select(integer_columns)

# Show the schemas of the new DataFrames
df_integer.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Descriptive Statistics:

# COMMAND ----------

display(df_integer.describe())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Missing Values:

# COMMAND ----------

columns_to_check = ['customer_name', 'VAT_number', 'credit_rating', 'years_in_business', 'TNW_in_MEUR', 'profits_perc_TNW', 'positive_working_capital', 'TNW_to_total_exposure', 'legal_form', 'fundation_year', 'fundation_month', 'fundation_day', 'financing_currency', 'fleet_size', 'fleet_own_trucks', 'fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 'fleet_vfs_EC', 'fleet_other_EC', 'Offer_number_1', 'Offer_number_2', 'Offer_number_3', 'financial_performance_currency', 'type_of_request', 'flag_domestic_use', 'vfs_customer', 'vfs_known_since', 'payment_discipline', 'total_exposure', 'quote_id', 'current_period', 'current_period_number_months', 'current_period_revenue', 'current_period_EBIT', 'current_period_depreciation', 'current_period_net_profit', 'current_period_fixed_assets', 'current_period_intangible_assets', 'current_period_current_assets', 'current_period_tangible_net_worth', 'current_period_long_term_liabilities', 'current_period_long_term_credit', 'current_period_short_term_liabilities', 'current_period_short_term_credit', 'current_period_off_balance_liabilities', 'last_period', 'last_period_number_months', 'last_period_revenue', 'last_period_EBIT', 'last_period_depreciation', 'last_period_net_profit', 'last_period_fixed_assets', 'last_period_intangible_assets', 'last_period_current_assets', 'last_period_tangible_net_worth', 'last_period_long_term_liabilities', 'last_period_long_term_credit', 'last_period_short_term_liabilities', 'last_period_short_term_credit', 'last_period_off_balance_liabilities', 'period_before_last_period', 'period_before_last_period_number_months', 'period_before_last_period_revenue', 'period_before_last_period_EBIT', 'period_before_last_period_depreciation', 'period_before_last_period_net_profit', 'period_before_last_period_fixed_assets', 'period_before_last_period_intangible_assets', 'period_before_last_period_current_assets', 'period_before_last_period_tangible_net_worth', 'period_before_last_period_long_term_liabilities', 'period_before_last_period_long_term_credit', 'period_before_last_period_short_term_liabilities', 'period_before_last_period_short_term_credit', 'period_before_last_period_off_balance_liabilities', 'type_of_data']

# Initialize an empty dictionary to store the null counts per column per market
null_counts_dict = {}

# Iterate over each column and calculate null counts
for column in columns_to_check:
    null_counts = df_new.groupBy("market").agg(
      F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(column)
    )
    
    # Collect the null counts in the dictionary
    null_counts_dict[column] = null_counts

# Sum all the null counts across columns for each market
total_sum_df = null_counts_dict[columns_to_check[0]]  # Initialize with the first column
for column in columns_to_check[1:]:
    total_sum_df = total_sum_df.join(null_counts_dict[column], on="market", how="inner")

# Cast the columns to integer before summing
for column in columns_to_check:
    total_sum_df = total_sum_df.withColumn(column, total_sum_df[column].cast("int"))

# Calculate the total sum and create a new column named 'total_sum'
total_sum_df = total_sum_df.withColumn("total_sum", reduce(lambda x, y: x + y, [total_sum_df[col] for col in columns_to_check]))

# Show the result
display(total_sum_df)



# COMMAND ----------

# Select and display only the 'market' and 'total_sum' columns
result_df = total_sum_df.select("market", "total_sum")
result_df.show()


# COMMAND ----------

# Collect data to the local environment
local_result_df = result_df.toPandas()

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(local_result_df['market'], local_result_df['total_sum'], color='blue')
plt.title('Total Null Counts per Market')
plt.xlabel('Market')
plt.ylabel('Total Null Counts')
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# COMMAND ----------

df_filtered = df.filter(df['market']== "BG")
display(df_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for possbile outliers in the numerical veriables

# COMMAND ----------

# Assuming df_new2 is your PySpark DataFrame
df_new2 = df_to_int

# List of numerical columns for EDA
columns_for_eda = ['current_period_revenue', 'current_period_EBIT', 'current_period_depreciation',
                   'current_period_net_profit', 'current_period_fixed_assets', 'current_period_intangible_assets',
                   'current_period_current_assets', 'current_period_tangible_net_worth',
                   'current_period_long_term_liabilities', 'current_period_long_term_credit',
                   'current_period_short_term_liabilities', 'current_period_short_term_credit',
                   'current_period_off_balance_liabilities', 'last_period_number_months', 'last_period_revenue',
                   'last_period_EBIT', 'last_period_depreciation', 'last_period_net_profit', 'last_period_fixed_assets',
                   'last_period_intangible_assets', 'last_period_current_assets', 'last_period_tangible_net_worth',
                   'last_period_long_term_liabilities', 'last_period_long_term_credit', 'last_period_short_term_liabilities',
                   'last_period_short_term_credit', 'last_period_off_balance_liabilities',
                   'period_before_last_period_number_months', 'period_before_last_period_revenue',
                   'period_before_last_period_EBIT', 'period_before_last_period_depreciation',
                   'period_before_last_period_net_profit', 'period_before_last_period_fixed_assets',
                   'period_before_last_period_intangible_assets', 'period_before_last_period_current_assets',
                   'period_before_last_period_tangible_net_worth', 'period_before_last_period_long_term_liabilities',
                   'period_before_last_period_long_term_credit', 'period_before_last_period_short_term_liabilities',
                   'period_before_last_period_short_term_credit', 'period_before_last_period_off_balance_liabilities']

# Select columns for EDA
eda_df = df_new2.select(columns_for_eda)

# Convert PySpark DataFrame to Pandas for plotting with seaborn
eda_df_pd = eda_df.toPandas()

# Plot boxplots for numerical variables
plt.figure(figsize=(15, 10))
sns.boxplot(data=eda_df_pd)
plt.title('Boxplot for Numerical Columns')
plt.xticks(rotation=90)
plt.ylabel('Values')
plt.show()

