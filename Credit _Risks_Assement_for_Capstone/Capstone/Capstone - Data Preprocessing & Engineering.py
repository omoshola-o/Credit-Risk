# Databricks notebook source
# MAGIC %md
# MAGIC # Install packages

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

# COMMAND ----------

# MAGIC %md
# MAGIC # Import data

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

# display(df)
# df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Unwanted columns

# COMMAND ----------

# drop unwanted columns
df1 = df.drop('VAT_number', 'legal_form', 'fundation_month', 'fundation_day', 'fundation_year', 'fleet_own_trucks','fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 'fleet_vfs_EC', 'fleet_other_EC', 'Offer_number_1', 'Offer_number_2', 'Offer_number_3','financial_performance_currency','type_of_request', 'flag_domestic_use','vfs_customer','vfs_known_since','quote_id', 'type_of_data')
# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop Unwanted records

# COMMAND ----------

# Some rows from the data has '0', 'null', 'nan' as the customer name, which did not represent any entity from the data. These items has no meaniningful variables accross the entire dataset. These rows will be dropped inorder to ascertain a meaningful sense of the data.

#dropping list of unwanted records from the data
drop_list = ['0', 'null', 'nan']

df1 = df1.filter(~col("customer_name").isin(drop_list))
# df1.display()

# COMMAND ----------

df1.columns
df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Column Names and DataType Processing 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Change Columns Names

# COMMAND ----------

# dictionary mapping original column names -> new names  
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

# perform rename
for current_name, new_name in renames.items():  
    df1 = df1.withColumnRenamed(current_name, new_name)

# show dataframe with new column names
df1.printSchema() 

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataType Converstion

# COMMAND ----------

# list of columns to convert to Double
columns_to_double = [
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
# convert specified columns to double
for column in columns_to_double:
    df1 = df1.withColumn(column, col(column).cast(DoubleType()))

# show the updated DataFrame
# df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Fill null values with mean values

# COMMAND ----------

# calculate mean values for numerical columns
impute_cols = [col for col, dtype in df1.dtypes if dtype == 'double']
means = df1.select([mean(col).alias(col) for col in impute_cols]).collect()[0].asDict()

# fill null values with mean values
for col_name in impute_cols:
    mean_value = means[col_name]
    df1 = df1.withColumn(col_name, when(col(col_name).isNull(), mean_value).otherwise(col(col_name)))

# show the updated DataFrame
# df1.display()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summation of periodical financial variable to one.

# COMMAND ----------

from pyspark.sql.functions import col, expr

# sum revenue across periods 
df1 = df1.withColumn("revenue", 
                   expr("CP_revenue + LP_revenue + PBLP_revenue"))

# sum EBIT
df1 = df1.withColumn("EBIT",  
                   expr("CP_EBIT + LP_EBIT + PBLP_EBIT"))

# sum depreciation
df1 = df1.withColumn("depreciation",  
                   expr("CP_depreciation + LP_depreciation + PBLP_depreciation"))

# sum Net Profit
df1 =  df1.withColumn("net_profit",
                   expr("CP_net_profit + LP_net_profit + PBLP_depreciation"))

# sum fixed asset
df1 = df1.withColumn("fixed_assets",
                   expr("CP_fixed_assets + LP_fixed_assets + PBLP_fixed_assets"))

# sum Intangible assets
df1 = df1.withColumn("intangible_assets",
                   expr("CP_intangible_assets + LP_intangible_assets + PBLP_intangible_assets"))

# sum current Assets
df1 = df1.withColumn("current_assets",
                   expr("CP_current_assets + LP_current_assets + PBLP_current_assets"))

# sum Tangible Networth
df1 = df1.withColumn("tangible_net_worth",
                   expr("CP_tangible_net_worth + LP_tangible_net_worth + PBLP_tangible_net_worth"))

# sum Long Term Liability
df1 = df1.withColumn("long_term_liab",
                   expr("CP_long_term_liab + LP_long_term_liab + PBLP_long_term_liab"))

# sum Long Term Credit
df1 = df1.withColumn("long_term_credit",
                   expr("CP_long_term_credit + LP_long_term_credit + PBLP_long_term_credit"))

# sum Short Term Liability
df1 = df1.withColumn("short_term_liab",
                   expr("CP_short_term_liab + LP_short_term_liab + PBLP_short_term_liab"))

# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapping Credit Rating to numerical values

# COMMAND ----------

columns = ["CR_Rating"]

# defining a mapping for credit ratings to numerical values
credit_rating_mapping = {"A": 100, "B": 75, "C": 50, "D": 25}

# Using the when and otherwise functions to create a new numerical column
df1 = df1.withColumn("CR_Rating_score", 
                   when(df1["CR_Rating"] == "A", credit_rating_mapping["A"])
                   .when(df1["CR_Rating"] == "B", credit_rating_mapping["B"])
                   .when(df1["CR_Rating"] == "C", credit_rating_mapping["C"])
                   .when(df1["CR_Rating"] == "D", credit_rating_mapping["D"])
                   .otherwise(None))

# showing the resulting DataFrame
# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapping payment discipline to numerical values

# COMMAND ----------

columns = ["pmt_discipline"]

# defining a mapping for payment discipline  to numerical values
payment_discipline_mapping = {"Excellent": 100, "Good": 75, "Fair": 50, "Bad": 25, "None": 25,}

# using the when and otherwise functions to create a new target_variable column
df1 = df1.withColumn("pmt_discipline_score", 
                   when(df1["pmt_discipline"] == "Excellent", payment_discipline_mapping["Excellent"])
                   .when(df1["pmt_discipline"] == "Good", payment_discipline_mapping["Good"])
                   .when(df1["pmt_discipline"] == "Fair", payment_discipline_mapping["Fair"])
                   .otherwise(25))

# show.
# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Engineering of Key Financial Ratios

# COMMAND ----------

# Debt to Equity Ratio
df1 = df1.withColumn(
    "debt_equity_ratio",
    expr(
      "(long_term_liab + short_term_liab) / (tangible_net_worth)"
    )
)
# Debt to Assets Ratio
df1 = df1.withColumn(
    "debt_asset_ratio",
    expr(
      "(long_term_liab + short_term_liab) / (fixed_assets + intangible_assets + current_assets)"
    )
)
# Current Ratio
df1 = df1.withColumn(
    "current_ratio",
    expr(
      "(current_assets) / (short_term_liab)"
    )
)
# Return on Assets (ROA)
df1 = df1.withColumn(
    "return_on_assets",
    expr(
      "(net_profit ) / (fixed_assets + intangible_assets + current_assets)"
    )
)

# show the updated DataFrame with the new column
# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Drop features where financial ratios were derived from

# COMMAND ----------

# droping unwanted columns
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
# display(df1)

# COMMAND ----------

# get column data types
schema = df1.schema

numerical_cols = [field.name for field in schema if isinstance(field.dataType,  DoubleType)]

categorical_cols = [field.name for field in schema if isinstance(field.dataType, StringType)] 

date_cols = [field.name for field in schema if isinstance(field.dataType, DateType)]

# display categorized columns
categorized_columns = {
  "Numerical Columns": numerical_cols,
  "Categorical Columns": categorical_cols,
  "Date Columns": date_cols,
  }

for dtype, cols in categorized_columns.items():
  print(dtype)
  print(cols)
  # print()

# COMMAND ----------

# MAGIC %md
# MAGIC # Seting Target Variable

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Assumption and Threshold for Current Ration for the definition of the target varible

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### Current Ratio formula:
# MAGIC ##### Current Assets / Current Liabilities
# MAGIC
# MAGIC ##### Check if the current ratio of the customer is at or above the 1.5x threshold. 
# MAGIC
# MAGIC ##### Assumption:
# MAGIC ##### - If the customer has $2 million in current assets.
# MAGIC ##### - And $1 million in current liabilities.
# MAGIC ##### - Current ratio = Current Assets / Current liabilities 
# MAGIC #####            = $2 million / $1 million  
# MAGIC #####            = 2x
# MAGIC
# MAGIC ##### This would meet the working capital adequacy benchmark.
# MAGIC
# MAGIC ##### If the current ratio was instead only 1.1x ($2 million current assets / $1.8 million current liabilities), then
# MAGIC ##### flag the company for further assessment of their liquidity to meet short-term obligations.
# MAGIC
# MAGIC ##### A minimum level of 1.5x could be reasonable threshold for adequate working capital.
# MAGIC

# COMMAND ----------

# Calculate current ratio 
df1 = df1.withColumn("current_ratio", col("current_assets")/col("short_term_liab"))

# Filter companies below 1.5x ratio
min_ratio = 1.5

# Print results 
# display(df1)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting risk category based on risk thresholds for target varible

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
# display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Checking for Correlation Coefiicient in the features

# COMMAND ----------


# Identify integer and float columns 
int_cols = [col for col, dtype in df1.dtypes if dtype in 'double']

# Calculate correlation matrix  
correlation_matrix = df1.select(int_cols).toPandas().corr()

# print("Correlation Matrix:")
# correlation_matrix.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rechecking for Clean Data

# COMMAND ----------

# Fill null values with zero
df1 = df1.fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC # Write clean data to Hive for Modelling

# COMMAND ----------

# List of columns to select

cols_to_select = ['customer_name',
 'TNW_in_MEUR',
 'profits_perc_TNW',
 'Positive_WC',
 'TNW_to_T-Exposure',
 'fleet_size',
 'total_exposure',
 'revenue',
 'EBIT',
 'depreciation',
 'net_profit',
 'fixed_assets',
 'intangible_assets',
 'current_assets',
 'tangible_net_worth',
 'long_term_liab',
 'long_term_credit',
 'short_term_liab',
 'CR_Rating_score',
 'pmt_discipline_score',
 'debt_equity_ratio',
 'debt_asset_ratio',
 'current_ratio',
 'return_on_assets',
 'target_variable']
   
df1 = df1.select(cols_to_select) 

# Print schema to verify columns
df1.printSchema()  

spark.sql('DROP TABLE IF EXISTS hive_metastore.default.crs_transformed')
df1.write.mode('overwrite').saveAsTable('hive_metastore.default.crs_transformed')

print("DataFrame written to Hive table successfully!")
