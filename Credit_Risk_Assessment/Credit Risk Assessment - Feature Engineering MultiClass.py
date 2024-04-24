# Databricks notebook source
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

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop Unwanted columns

# COMMAND ----------

# Drop unwanted columns
df1 = df.drop('VAT_number', 'legal_form', 'fundation_month', 'fundation_day', 'fleet_own_trucks','fleet_vfs_trucks', 'fleet_other_trucks', 'fleet_own_EC', 
         'fleet_vfs_EC', 'fleet_other_EC', 'Offer_number_1', 'Offer_number_2', 'Offer_number_3','financial_performance_currency',
         'type_of_request', 'flag_domestic_use','vfs_customer','vfs_known_since','quote_id', 'type_of_data')
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Numerical Data to IntegerType format and Fill Null Values

# COMMAND ----------

# List of columns to convert to IntegerType
columns_to_integer = [
    "years_in_business", "TNW_in_MEUR", "profits_perc_TNW", "positive_working_capital",
    "TNW_to_total_exposure", "fleet_size", "total_exposure", "current_period_number_months",
    "current_period_revenue", "current_period_EBIT", "current_period_depreciation",
    "current_period_net_profit", "current_period_fixed_assets", "current_period_intangible_assets", 
    "current_period_current_assets", "current_period_tangible_net_worth","current_period_long_term_liabilities",
    "current_period_long_term_credit", "current_period_short_term_liabilities","current_period_short_term_credit",
    "current_period_off_balance_liabilities", "last_period_number_months","last_period_revenue",
    "last_period_EBIT", "last_period_depreciation", "last_period_net_profit", "last_period_fixed_assets",
    "last_period_intangible_assets", "last_period_current_assets", "last_period_tangible_net_worth", "last_period_long_term_liabilities",
    "last_period_long_term_credit", "last_period_short_term_liabilities", "last_period_short_term_credit", "last_period_off_balance_liabilities","period_before_last_period_number_months", "period_before_last_period_revenue", "period_before_last_period_EBIT",
    "period_before_last_period_depreciation", "period_before_last_period_net_profit", "period_before_last_period_fixed_assets", "period_before_last_period_intangible_assets",
    "period_before_last_period_current_assets", "period_before_last_period_tangible_net_worth", "period_before_last_period_long_term_liabilities","period_before_last_period_long_term_credit", "period_before_last_period_short_term_liabilities", "period_before_last_period_short_term_credit",
    "period_before_last_period_off_balance_liabilities"
]

# Convert specified columns to IntegerType
for column in columns_to_integer:
    df1 = df1.withColumn(column, col(column).cast(DoubleType()))

# Replace null values with 0 in all columns
df1 = df1.fillna(0)

# # Drops the data having null values 
# df1 = df1.na.drop() 

# Show the updated DataFrame
df1.printSchema()
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Date Data to DateType format

# COMMAND ----------

columns_to_date = [
    "fundation_year", "current_period",
    "last_period", "period_before_last_period",
]

# Convert specified columns to DateType
for column in columns_to_date:
    df1 = df1.withColumn(column, col(column).cast(DateType()))

# Show the updated DataFrame
df1.printSchema()
display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check if the Ramaining Null Values Does not Affect Key Numerical Data

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


# Dictionary mapping original -> new names  
renames = {
"customer_name": "customer_name",
"VAT_number": "VAT_Num",
"credit_rating": "CR_Rating",
"years_in_business": "YRS_In_BUS",
"TNW_in_MEUR": "TNW_in_MEUR",
"profits_perc_TNW": "profits_perc_TNW",
"positive_working_capital": "Positive_WC",
"TNW_to_total_exposure": "TNW_to_T-Exposure",
"legal_form": "legal_form",
"fundation_year": "Date_of_Est. ",
"fundation_month": "Date_of_Est. ",
"fundation_day": "Date_of_Est.",
"financing_currency": "financing_currency",
"fleet_size": "fleet_size",
"fleet_own_trucks": "fleet_size",
"fleet_vfs_trucks": "fleet_size",
"fleet_other_trucks": "fleet_size",
"fleet_own_EC": "fleet_size",
"fleet_vfs_EC": "fleet_size",
"fleet_other_EC": "fleet_size",
"Offer_number_1": "Offer_numer",
"Offer_number_2": "Offer_numer",
"Offer_number_3": "Offer_numer",
"financial_performance_currency": "constant_currency",
"type_of_request": "type_of_request",
"flag_domestic_use": "flag_domestic_use",
"vfs_customer": "vfs_customer",
"vfs_known_since": "vfs_known_since",
"payment_discipline": "pmt_discipline",
"total_exposure": "total_exposure",
"quote_id": "quote_id",
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
"type_of_data": "type_of_data" 
}

# Perform rename
for current_name, new_name in renames.items():  
    df1 = df1.withColumnRenamed(current_name, new_name)

# Show dataframe with new column names
df1.printSchema() 

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

df1.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering

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
payment_discipline_mapping = {"Excellent": 100, "Good": 75, "Fair": 50, "Bad": 25,"None": 0,}

# Use the when and otherwise functions to create a new target_variable column
df1 = df1.withColumn("pmt_discipline_score", 
                   when(df1["pmt_discipline"] == "Excellent", payment_discipline_mapping["Excellent"])
                   .when(df1["pmt_discipline"] == "Good", payment_discipline_mapping["Good"])
                   .when(df1["pmt_discipline"] == "Fair", payment_discipline_mapping["Fair"])
                   .when(df1["pmt_discipline"] == "Bad", payment_discipline_mapping["Bad"])
                   .when(df1["pmt_discipline"] == "None", payment_discipline_mapping["None"])                   
                   .otherwise(None))

# change the target variable to results from risk threshold.


# COMMAND ----------

# MAGIC %md
# MAGIC ### Debt-Risk Ratios
# MAGIC

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
# Replace null values with 0 in all columns
df1 = df1.fillna(0)
# Show the updated DataFrame with the new column
display(df1)

# COMMAND ----------

df1.printSchema()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors  

spark = SparkSession.builder.appName('CorrelationMatrix').getOrCreate()

# List of column names
cols = [
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

# Create sample data frame with column names  
import numpy as np
num_rows = 50
data = np.random.rand(num_rows, len(cols)) 
# df = spark.createDataFrame(data, cols)  

assembler = VectorAssembler(inputCols=cols,outputCol="features")
df1_vector = assembler.transform(df1)

matrix = Correlation.corr(df1_vector, "features").head()[0].toArray()

# import pandas as pd
# pd.DataFrame(matrix, columns=cols, index = cols)["target_variable"].sort_values()


import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure(figsize=(20, 15))
sns.set(font_scale=1)  
heatmap = sns.heatmap(np.array(matrix), annot=True, 
                    vmin=-1, vmax=1, xticklabels=cols, yticklabels=cols) 

plt.xticks(rotation=90) 
plt.yticks(rotation=0)  

plt.tight_layout()
plt.show()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

# Create a Spark session
spark = SparkSession.builder.appName("RiskCategories").getOrCreate()

# Define risk thresholds
low_risk_threshold = 50
medium_risk_threshold_low = 25
medium_risk_threshold_high = 50
high_risk_threshold_low = 25

# Create a new column 'risk_category' based on risk thresholds
df1 = df1.withColumn(
    "risk_category",
    when(
        (col("profits_perc_TNW") > low_risk_threshold) &
        (col("TNW_to_T-Exposure") > low_risk_threshold) &
        (col("CR_Rating_score") > 75) &
        (col("pmt_discipline_score") > 75) &
        (col("debt_equity_ratio") > low_risk_threshold) &
        (col("debt_asset_ratio") > low_risk_threshold) &
        (col("interest_cov_ratio") > low_risk_threshold) &
        (col("current_ratio") > low_risk_threshold) &
        (col("return_on_assets") > low_risk_threshold) &
        (col("debt_ser_cov_ratio") > low_risk_threshold),
        "Low"
    ).when(
        (col("profits_perc_TNW").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("TNW_to_T-Exposure").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("CR_Rating_score").between(50, 75)) &
        (col("pmt_discipline_score").between(50, 75)) &
        (col("debt_equity_ratio").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("debt_asset_ratio").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("interest_cov_ratio").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("current_ratio").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("return_on_assets").between(medium_risk_threshold_low, medium_risk_threshold_high)) &
        (col("debt_ser_cov_ratio").between(medium_risk_threshold_low, medium_risk_threshold_high)),
        "Medium"
    # ).when(
    #     (col("profits_perc_TNW") < high_risk_threshold_low) &
    #     (col("TNW_to_T-Exposure") > low_risk_threshold) &
    #     (col("CR_Rating_score") < 50) &
    #     (col("pmt_discipline_score") < 50) &
    #     (col("debt_equity_ratio") < high_risk_threshold_low) &
    #     (col("debt_asset_ratio") < high_risk_threshold_low) &
    #     (col("interest_cov_ratio") < high_risk_threshold_low) &
    #     (col("current_ratio") < high_risk_threshold_low) &
    #     (col("return_on_assets") < high_risk_threshold_low) &
    #     (col("debt_ser_cov_ratio") < high_risk_threshold_low),
    #     "High"
    ).otherwise("High")
)

# Show the resulting DataFrame with the new 'risk_category' column
display(df1)

# COMMAND ----------

# #After necessary financial ratios have been engineered

# from pyspark.sql.functions import when, col

# # Sample dataframe with financial data
# # df = spark.createDataFrame([
# # risk definition as per columns

# # profits_perc_TNW < 10%
# # TNW_to_T-Exposure > 1.5
# # short_term_leverage < 0.1

# # Medium risk:
# # 10% <= profits_perc_TNW <= 25%
# # 1.0 <= TNW_to_T-Exposure <= 1.5
# # 0.1 <= short_term_leverage <= 0.25

# # High risk:
# # profits_perc_TNW > 25%
# # TNW_to_T-Exposure < 1.0
# # short_term_leverage > 0.25

# # Where:
# # profits_perc_TNW: Profitability as a percentage of tangible net worth
# # TNW_to_T-Exposure: Tangible net worth coverage of total exposure
# # # short_term_leverage: Ratio of short term liabilities to equity
# # ])

# # Add risk threshold columns
# df1 = df1.withColumn("risk", 
#     when((col("profits_perc_TNW") < ?) &  
#          (col("TNW_to_T-Exposure") > ?) &
#          (col("short_term_leverage") < ?), "low")
#     .when((col("profits_perc_TNW").between(0.1, 0.25)) & 
#           (col("TNW_to_T-Exposure").between(1.0, 1.5)) & 
#           (col("short_term_leverage").between(0.1, 0.25)), "medium")
#     .otherwise("high")
# )

# # # Filter by risk threshold if needed
# # low_risk_df = df.filter(col("risk") == "low")

# # # Save transformed dataframe
# # df.write.parquet("risk_transformed.parquet")

# # Show the resulting DataFrame
# display(df1)

# COMMAND ----------

display(df1.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ###  Risk Threshold

# COMMAND ----------


# Low Risk:
# - profits_perc_TNW > 50%
# - TNW_to_T-Exposure > 50%
# - CR_Rating_score > 75%
# - pmt_discipline_score > 75%
# - debt_equity_ratio > 50%
# - debt_asset_ratio > 50%
# - interest_cov_ratio > 50%
# - current_ratio > 50%
# - return_on_assets > 50%
# - debt_ser_cov_ratio > 50%

# Medium Risk:
# - 50% <= profits_perc_TNW >= 25%
# - 50% <= TNW_to_T-Exposure >= 25%
# - 75% <= CR_Rating_score >= 50%
# - 75% <= pmt_discipline_score >= 50%
# - 50% <= debt_equity_ratio >= 25%
# - 50% <= debt_asset_ratio >= 25%
# - 50% <= interest_cov_ratio >= 25%
# - 50% <= current_ratio >= 25%
# - 50% <= return_on_assets >= 25%
# - 50% <= debt_ser_cov_ratio >= 25%

# High Risk:
# - profits_perc_TNW < 25%
# - TNW_to_T-Exposure > 25%
# - CR_Rating_score < 50%
# - pmt_discipline_score < 50%
# - debt_equity_ratio < 25%
# - debt_asset_ratio < 25%
# - interest_cov_ratio < 25% 
# - current_ratio < 25%
# - return_on_assets < 25%
# - debt_ser_cov_ratio < 25%

# COMMAND ----------

# create a column that has low, medium, and high risk, where
# # Low Risk is
# # - profits_perc_TNW > 50% of profits_perc_TNW
# # - TNW_to_T-Exposure > 50% of TNW_to_T-Exposure
# such that when (col("profits_perc_TNW") > 50% and all other low risk threshold are met
# and apply the same principle accross the below otherwise("high")
# # - CR_Rating_score > 75%
# # - pmt_discipline_score > 75%
# # - debt_equity_ratio > 50%
# # - debt_asset_ratio > 50%
# # - interest_cov_ratio > 50%
# # - current_ratio > 50%
# # - return_on_assets > 50%
# # - debt_ser_cov_ratio > 50%

# # Medium Risk:
# # - 50% <= profits_perc_TNW >= 25%
# # - 50% <= TNW_to_T-Exposure >= 25%
# # - 75% <= CR_Rating_score >= 50%
# # - 75% <= pmt_discipline_score >= 50%
# # - 50% <= debt_equity_ratio >= 25%
# # - 50% <= debt_asset_ratio >= 25%
# # - 50% <= interest_cov_ratio >= 25%
# # - 50% <= current_ratio >= 25%
# # - 50% <= return_on_assets >= 25%
# # - 50% <= debt_ser_cov_ratio >= 25%

# # High Risk:
# # - profits_perc_TNW < 25%
# # - TNW_to_T-Exposure > 25%
# # - CR_Rating_score < 50%
# # - pmt_discipline_score < 50%
# # - debt_equity_ratio < 25%
# # - debt_asset_ratio < 25%
# # - interest_cov_ratio < 25% 
# # - current_ratio < 25%
# # - return_on_assets < 25%
# # - debt_ser_cov_ratio < 25% 

# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import when, col

# # Create a Spark session
# spark = SparkSession.builder.appName("RiskAnalysis").getOrCreate()

# # Assuming you have a DataFrame named 'df' with columns mentioned in the instructions

# # Define the conditions for each risk level
# low_risk_conditions = [
#     (col("profits_perc_TNW") > 0.5),
#     (col("TNW_to_T-Exposure") > 0.5),
#     (col("CR_Rating_score") > 0.75),
#     (col("pmt_discipline_score") > 0.75),
#     (col("debt_equity_ratio") > 0.5),
#     (col("debt_asset_ratio") > 0.5),
#     (col("interest_cov_ratio") > 0.5),
#     (col("current_ratio") > 0.5),
#     (col("return_on_assets") > 0.5),
#     (col("debt_ser_cov_ratio") > 0.5)
# ]

# medium_risk_conditions = [
#     (col("profits_perc_TNW").between(0.25, 0.5)),
#     (col("TNW_to_T-Exposure").between(0.25, 0.5)),
#     (col("CR_Rating_score").between(0.5, 0.75)),
#     (col("pmt_discipline_score").between(0.5, 0.75)),
#     (col("debt_equity_ratio").between(0.25, 0.5)),
#     (col("debt_asset_ratio").between(0.25, 0.5)),
#     (col("interest_cov_ratio").between(0.25, 0.5)),
#     (col("current_ratio").between(0.25, 0.5)),
#     (col("return_on_assets").between(0.25, 0.5)),
#     (col("debt_ser_cov_ratio").between(0.25, 0.5))
# ]

# high_risk_conditions = [
#     (col("profits_perc_TNW") < 0.25),
#     (col("TNW_to_T-Exposure") > 0.25),
#     (col("CR_Rating_score") < 0.5),
#     (col("pmt_discipline_score") < 0.5),
#     (col("debt_equity_ratio") < 0.25),
#     (col("debt_asset_ratio") < 0.25),
#     (col("interest_cov_ratio") < 0.25),
#     (col("current_ratio") < 0.25),
#     (col("return_on_assets") < 0.25),
#     (col("debt_ser_cov_ratio") < 0.25)
# ]

# # Assign the risk level based on the conditions
# df1 = df1.withColumn("Risk_Level",
#     when(reduce(lambda x, y: x & y, low_risk_conditions), "Low Risk")
#     .when(reduce(lambda x, y: x & y, medium_risk_conditions), "Medium Risk")
#     .when(reduce(lambda x, y: x & y, high_risk_conditions), "High Risk")
#     .otherwise("Unknown")
# )

# # Show the DataFrame with the new "Risk_Level" column
# display(df1)


# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, sum, when

# # Create a Spark session
# spark = SparkSession.builder.appName("RiskAnalysis").getOrCreate()

# # Assuming you have a DataFrame named 'df'

# # Define the list of columns to calculate the sum and apply risk thresholds
# columns_to_sum = [
#     "YRS_In_BUS",
#     "TNW_in_MEUR",
#     "profits_perc_TNW",
#     "Positive_WC",
#     "TNW_to_T-Exposure",
#     "fleet_size",
#     "total_exposure",
#     "revenue",
#     "EBIT",
#     "depreciation",
#     "net_profit",
#     "fixed_assets",
#     "intangible_assets",
#     "current_assets",
#     "tangible_net_worth",
#     "long_term_liab",
#     "long_term_credit",
#     "short_term_liab",
#     "short_term_credit",
#     "off_balance_liab",
#     "CR_Rating_score",
#     "pmt_discipline_score",
#     "debt_equity_ratio",
#     "debt_asset_ratio",
#     "interest_cov_ratio",
#     "current_ratio",
#     "return_on_assets",
#     "debt_ser_cov_ratio"
# ]

# # Calculate the sum of values in each specified column
# column_sums = df1.agg(*(sum(col(c)).alias(c) for c in columns_to_sum)).collect()[0]

# # Calculate the percentage for each cell in each specified column
# df1
# for column_name in columns_to_sum:
#     df1 = df1.withColumn(column_name + "_percentage", (col(column_name) / column_sums[column_name]) * 100)

# # Define the risk thresholds as a percentage of values in each specified column
# low_risk_percentage = 50
# medium_risk_percentage = 25
# high_risk_percentage = 25

# # Calculate the overall risk level based on the conditions in each specified column
# df1 = df1.withColumn(
#     "Risk_Level",
#     when(reduce(lambda x, y: x & y, [col(c + "_percentage") > low_risk_percentage for c in columns_to_sum]), "Low Risk")
#     .when(reduce(lambda x, y: x & y, [col(c + "_percentage").between(medium_risk_percentage, low_risk_percentage) for c in columns_to_sum]), "Medium Risk")
#     .when(reduce(lambda x, y: x & y, [col(c + "_percentage") < high_risk_percentage for c in columns_to_sum]), "High Risk")
#     .otherwise("Unknown")
# )

# # Show the DataFrame with the new "Risk_Level" column
# display(df1)


# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, sum, when

# # Create a Spark session
# spark = SparkSession.builder.appName("RiskAnalysis").getOrCreate()

# # Define the list of columns to calculate the sum and apply risk thresholds
# columns_to_sum = [
#     "YRS_In_BUS",
#     "TNW_in_MEUR",
#     "profits_perc_TNW",
#     "Positive_WC",
#     "TNW_to_T-Exposure",
#     "fleet_size",
#     "total_exposure",
#     "revenue",
#     "EBIT",
#     "depreciation",
#     "net_profit",
#     "fixed_assets",
#     "intangible_assets",
#     "current_assets",
#     "tangible_net_worth",
#     "long_term_liab",
#     "long_term_credit",
#     "short_term_liab",
#     "short_term_credit",
#     "off_balance_liab",
#     "CR_Rating_score",
#     "pmt_discipline_score",
#     "debt_equity_ratio",
#     "debt_asset_ratio",
#     "interest_cov_ratio",
#     "current_ratio",
#     "return_on_assets",
#     "debt_ser_cov_ratio"
# ]

# # Calculate the sum of values in each specified column
# column_sums = df1.agg(*(sum(col(c)).alias(c) for c in columns_to_sum)).collect()[0]

# # Calculate the percentage for each cell in each specified column
# df_percentage = df1
# for column_name in columns_to_sum:
#     df_percentage = df_percentage.withColumn(column_name + "_percentage", (col(column_name) / column_sums[column_name]) * 100)

# # Define the risk thresholds as a percentage of values in each specified column
# low_risk_percentage = 50
# medium_risk_percentage = 25
# high_risk_percentage = 25

# # Define conditions for each risk level in each specified column
# low_risk_conditions = [col(c + "_percentage") > low_risk_percentage for c in columns_to_sum]
# medium_risk_conditions = [col(c + "_percentage").between(medium_risk_percentage, low_risk_percentage) for c in columns_to_sum]
# high_risk_conditions = [col(c + "_percentage") < high_risk_percentage for c in columns_to_sum]

# # Assign the risk level based on the conditions in each specified column
# df_result = df_percentage
# for column_name in columns_to_sum:
#     df_result = df_result.withColumn(
#         column_name + "_Risk_Level",
#         when(low_risk_conditions[columns_to_sum.index(column_name)], "Low Risk")
#         .when(medium_risk_conditions[columns_to_sum.index(column_name)], "Medium Risk")
#         .when(high_risk_conditions[columns_to_sum.index(column_name)], "High Risk")
#         .otherwise("Unknown")
#     )

# # Show the DataFrame with the new "Risk_Level" columns
# display(df_result)


# COMMAND ----------

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, sum

# # Create a Spark session
# spark = SparkSession.builder.appName("RiskAnalysis").getOrCreate()

# # Define the list of columns to calculate the sum
# columns_to_sum = [
#     "YRS_In_BUS",
#     "TNW_in_MEUR",
#     "profits_perc_TNW",
#     "Positive_WC",
#     "TNW_to_T-Exposure",
#     "fleet_size",
#     "total_exposure",
#     "revenue",
#     "EBIT",
#     "depreciation",
#     "net_profit",
#     "fixed_assets",
#     "intangible_assets",
#     "current_assets",
#     "tangible_net_worth",
#     "long_term_liab",
#     "long_term_credit",
#     "short_term_liab",
#     "short_term_credit",
#     "off_balance_liab",
#     "CR_Rating_score",
#     "pmt_discipline_score",
#     "debt_equity_ratio",
#     "debt_asset_ratio",
#     "interest_cov_ratio",
#     "current_ratio",
#     "return_on_assets",
#     "debt_ser_cov_ratio"
# ]

# # Calculate the sum of specified columns
# column_sums = df1.agg(*(sum(col(c)).alias(c) for c in columns_to_sum)).collect()[0]

# # Print the column sums
# for column_name in columns_to_sum:
#     print(f"Sum of {column_name}: {column_sums[column_name]}")


# # Define the conditions for each risk level based on the column sums
# low_risk_conditions = [
#     (col("profits_perc_TNW") > 0.5 * column_sums["profits_perc_TNW"]),
#     (col("TNW_to_T-Exposure") > 0.5 * column_sums["TNW_to_T-Exposure"]),
#     (col("CR_Rating_score") > 0.75 * column_sums["CR_Rating_score"]),
#     (col("pmt_discipline_score") > 0.75 * column_sums["pmt_discipline_score"]),
#     (col("debt_equity_ratio") > 0.5 * column_sums["debt_equity_ratio"]),
#     (col("debt_asset_ratio") > 0.5 * column_sums["debt_asset_ratio"]),
#     (col("interest_cov_ratio") > 0.5 * column_sums["interest_cov_ratio"]),
#     (col("current_ratio") > 0.5 * column_sums["current_ratio"]),
#     (col("return_on_assets") > 0.5 * column_sums["return_on_assets"]),
#     (col("debt_ser_cov_ratio") > 0.5 * column_sums["debt_ser_cov_ratio"])
# ]

# medium_risk_conditions = [
#     (col("profits_perc_TNW") > 0.5 * column_sums["profits_perc_TNW"]),
#     (col("TNW_to_T-Exposure") > 0.5 * column_sums["TNW_to_T-Exposure"]),
#     (col("CR_Rating_score").between(0.5 * column_sums["CR_Rating_score"], 0.75 * column_sums["CR_Rating_score"])),
#     (col("pmt_discipline_score").between(0.5 * column_sums["pmt_discipline_score"], 0.75 * column_sums["pmt_discipline_score"])),
#     (col("debt_equity_ratio").between(0.25 * column_sums["debt_equity_ratio"], 0.5 * column_sums["debt_equity_ratio"])),
#     (col("debt_asset_ratio").between(0.25 * column_sums["debt_asset_ratio"], 0.5 * column_sums["debt_asset_ratio"])),
#     (col("interest_cov_ratio").between(0.25 * column_sums["interest_cov_ratio"], 0.5 * column_sums["interest_cov_ratio"])),
#     (col("current_ratio").between(0.25 * column_sums["current_ratio"], 0.5 * column_sums["current_ratio"])),
#     (col("return_on_assets").between(0.25 * column_sums["return_on_assets"], 0.5 * column_sums["return_on_assets"])),
#     (col("debt_ser_cov_ratio").between(0.25 * column_sums["debt_ser_cov_ratio"], 0.5 * column_sums["debt_ser_cov_ratio"]))
# ]

# high_risk_conditions = [
#     (col("profits_perc_TNW") > 0.5 * column_sums["profits_perc_TNW"]),
#     (col("TNW_to_T-Exposure") > 0.5 * column_sums["TNW_to_T-Exposure"]),
#     (col("CR_Rating_score") < 0.5 * column_sums["CR_Rating_score"]),
#     (col("pmt_discipline_score") < 0.5 * column_sums["pmt_discipline_score"]),
#     (col("debt_equity_ratio") < 0.25 * column_sums["debt_equity_ratio"]),
#     (col("debt_asset_ratio") < 0.25 * column_sums["debt_asset_ratio"]),
#     (col("interest_cov_ratio") < 0.25 * column_sums["interest_cov_ratio"]),
#     (col("current_ratio") < 0.25 * column_sums["current_ratio"]),
#     (col("return_on_assets") < 0.25 * column_sums["return_on_assets"]),
#     (col("debt_ser_cov_ratio") < 0.25 * column_sums["debt_ser_cov_ratio"])
# ]

# # Assign the risk level based on the conditions
# df1 = df1.withColumn("Risk_Level",
#     when(reduce(lambda x, y: x & y, low_risk_conditions), "Low Risk")
#     .when(reduce(lambda x, y: x & y, medium_risk_conditions), "Medium Risk")
#     .when(reduce(lambda x, y: x & y, high_risk_conditions), "High Risk")
#     .otherwise("Unknown")
# )

# # Show the DataFrame with the new "Risk_Level" column
# display(df1)


# COMMAND ----------

# from pyspark.sql.functions import when, col, expr, lit

# # Threshold percentiles 
# low_threshold = 0.50
# medium_threshold = 0.25

# conditions = [
#   (col("profits_perc_TNW").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("profits_perc_TNW")) &
#   (col("TNW_to_T-Exposure").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("TNW_to_T-Exposure")) & 
#   (col("CR_Rating_score").desc_nulls_last().cast("float").percentile(lit(0.75)) <= col("CR_Rating_score")) &  
#   (col("pmt_discipline_score").desc_nulls_last().cast("float").percentile(lit(0.75)) <= col("pmt_discipline_score")) &
#   (col("debt_equity_ratio").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("debt_equity_ratio")) & 
#   (col("debt_asset_ratio").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("debt_asset_ratio")) &
#   (col("interest_cov_ratio").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("interest_cov_ratio")) &
#   (col("current_ratio").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("current_ratio")) &
#   (col("return_on_assets").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("return_on_assets")) &
#   (col("debt_ser_cov_ratio").desc_nulls_last().cast("float").percentile(lit(low_threshold)) <= col("debt_ser_cov_ratio"))
# ]

# df1 = df1.withColumn("risk",
#     when(reduce(lambda x, y: x & y, conditions), lit("Low"))
#     .when(
#         (expr(""" 
#            profits_perc_TNW >= profits_perc_TNW.desc_nulls_last().cast("float").percentile(lit({0}))  
#            AND  
#            profits_perc_TNW <= profits_perc_TNW.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) &

#         (expr("""
#            TNW_to_T-Exposure >= TNW_to_T-Exposure.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            TNW_to_T-Exposure <= TNW_to_T-Exposure.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) &
       
#         (expr("""
#            CR_Rating_score >= CR_Rating_score.desc_nulls_last().cast("float").percentile(lit({0}))  
#            AND
#            CR_Rating_score <= CR_Rating_score.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(0.50, 0.75)) &
           
#         (expr("""
#            pmt_discipline_score >= pmt_discipline_score.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND  
#            pmt_discipline_score <= pmt_discipline_score.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(0.50, 0.75)) &

#         (expr("""
#            debt_equity_ratio >= debt_equity_ratio.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            debt_equity_ratio <= debt_equity_ratio.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) &       

#         (expr("""
#            debt_asset_ratio >= debt_asset_ratio.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            debt_asset_ratio <= debt_asset_ratio.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) & 

#         (expr("""
#            interest_cov_ratio >= interest_cov_ratio.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            interest_cov_ratio <= interest_cov_ratio.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) & 

#         (expr("""
#            current_ratio >= current_ratio.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            current_ratio <= current_ratio.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) &

#         (expr("""
#            return_on_assets >= return_on_assets.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            return_on_assets <= return_on_assets.desc_nulls_last().cast("float").percentile(lit({1}))
#         """).format(medium_threshold, low_threshold)) &

#         (expr("""
#            debt_ser_cov_ratio >= debt_ser_cov_ratio.desc_nulls_last().cast("float").percentile(lit({0}))
#            AND 
#            debt_ser_cov_ratio <= debt_ser_cov_ratio.desc_nulls_last().cast("float").percentile(lit({1}))
#     """).format(medium_threshold, low_threshold)) & lit("Medium"))
# )

# COMMAND ----------

# from pyspark.sql.functions import when, col, expr, lit
# from functools import reduce

# # Threshold percentiles
# low_threshold = 0.50
# medium_threshold = 0.25

# conditions = [
#     (col("profits_perc_TNW").desc_nulls_last().cast("float").percentile(0.50) <= col("profits_perc_TNW")) &
#     col("TNW_to_T-Exposure").desc_nulls_last().cast("float").percentile(0.50) <= col("TNW_to_T-Exposure")) &
#     (col("CR_Rating_score").desc_nulls_last().cast("float").percentile(0.75) <= col("CR_Rating_score")) &
#     (col("pmt_discipline_score").desc_nulls_last().cast("float").percentile(0.75) <= col("pmt_discipline_score")) &
#     (col("debt_equity_ratio").desc_nulls_last().cast("float").percentile(0.50) <= col("debt_equity_ratio")) &
#     (col("debt_asset_ratio").desc_nulls_last().cast("float").percentile(0.50) <= col("debt_asset_ratio")) &
#     (col("interest_cov_ratio").desc_nulls_last().cast("float").percentile(0.50) <= col("interest_cov_ratio")) &
#     (col("current_ratio").desc_nulls_last().cast("float").percentile(0.50) <= col("current_ratio")) &
#     (col("return_on_assets").desc_nulls_last().cast("float").percentile(0.50) <= col("return_on_assets")) &
#     (col("debt_ser_cov_ratio").desc_nulls_last().cast("float").percentile(low_threshold) <= col("debt_ser_cov_ratio"))
# ]

# # Import missing reduce function
# from functools import reduce

# df1 = df1.withColumn("risk",
#                     when(reduce(lambda x, y: x & y, conditions), lit("Low"))
#                     .when(
#                         (expr("""
#                            profits_perc_TNW >= percentile(profits_perc_TNW, {}) AND
#                            profits_perc_TNW <= percentile(profits_perc_TNW, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            TNW_to_T-Exposure >= percentile(TNW_to_T-Exposure, {}) AND
#                            TNW_to_T-Exposure <= percentile(TNW_to_T-Exposure, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            CR_Rating_score >= percentile(CR_Rating_score, {}) AND
#                            CR_Rating_score <= percentile(CR_Rating_score, {})
#                         """.format(0.50, 0.75))) &

#                         (expr("""
#                            pmt_discipline_score >= percentile(pmt_discipline_score, {}) AND
#                            pmt_discipline_score <= percentile(pmt_discipline_score, {})
#                         """.format(0.50, 0.75))) &

#                         (expr("""
#                            debt_equity_ratio >= percentile(debt_equity_ratio, {}) AND
#                            debt_equity_ratio <= percentile(debt_equity_ratio, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            debt_asset_ratio >= percentile(debt_asset_ratio, {}) AND
#                            debt_asset_ratio <= percentile(debt_asset_ratio, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            interest_cov_ratio >= percentile(interest_cov_ratio, {}) AND
#                            interest_cov_ratio <= percentile(interest_cov_ratio, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            current_ratio >= percentile(current_ratio, {}) AND
#                            current_ratio <= percentile(current_ratio, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            return_on_assets >= percentile(return_on_assets, {}) AND
#                            return_on_assets <= percentile(return_on_assets, {})
#                         """.format(medium_threshold, low_threshold))) &

#                         (expr("""
#                            debt_ser_cov_ratio >= percentile(debt_ser_cov_ratio, {}) AND
#                            debt_ser_cov_ratio <= percentile(debt_ser_cov_ratio, {})
#                         """.format(medium_threshold, low_threshold))) & lit("Medium"))
#                     )

# # Assuming you have already imported the required functions and have a PySpark session initialized.


# COMMAND ----------



# COMMAND ----------

# # add risk threshold columns 
# df1 = df1.withColumn("risk",
#     when( (col("CR_Rating_score") > 75) &
#           (col("pmt_discipline_score") > 75) &
#           (col("debt_equity_ratio") < 0.5) & 
#           (col("debt_asset_ratio") < 0.1) &
#           (col("interest_cov_ratio") > 5) &
#           (col("current_ratio") > 2) &  
#           (col("return_on_assets") > 0.1) &
#           (col("debt_ser_cov_ratio") > 1.5), "low")
#     .when( (col("CR_Rating_score").between(50, 75)) &
#            (col("pmt_discipline_score").between(50, 75)) &
#            (col("debt_equity_ratio").between(0.5, 1.0)) &
#            (col("debt_asset_ratio").between(0.1, 0.25)) &
#            (col("interest_cov_ratio").between(3, 5)) &
#            (col("current_ratio").between(1.5, 2)) &
#            (col("return_on_assets").between(0.05, 0.1)) &
#            (col("debt_ser_cov_ratio").between(1.2, 1.5)), "medium")
#     .otherwise("high")
# )

# COMMAND ----------

from pyspark.sql.functions import udf

@udf(returnType=IntegerType())  
def risk_to_num(risk):
    if risk == "low":
        return 0
    elif risk == "medium":
        return 1
    else:
        return 2

df1 = df1.withColumn("risk_num", risk_to_num("risk"))
display(df1)

# COMMAND ----------

# # List of columns to select

# cols_to_select = ["customer_name",
# "CR_Rating",
# "YRS_In_BUS",
# "TNW_in_MEUR",
# "profits_perc_TNW",
# "Positive_WC",
# "TNW_to_T-Exposure",
# "financing_currency",
# "fleet_size",
# "pmt_discipline",
# "total_exposure",
# "revenue",
# "EBIT",
# "depreciation",
# "net_profit",
# "fixed_assets",
# "intangible_assets",
# "current_assets",
# "tangible_net_worth",
# "long_term_liab",
# "long_term_credit",
# "short_term_liab",
# "short_term_credit",
# "off_balance_liab",
# "CR_Rating_score",
# "pmt_discipline_score",
# "debt_equity_ratio",
# "debt_asset_ratio",
# "interest_cov_ratio",
# "current_ratio",
# "return_on_assets",
# "debt_ser_cov_ratio",
# "risk",
# "risk_num"]
   
# df1 = df1.select(cols_to_select) 

# # Print schema to verify columns
# df1.printSchema()  

# spark.sql('DROP TABLE IF EXISTS hive_metastore.default.crs_transformed')
# df1.write.mode('overwrite').saveAsTable('hive_metastore.default.crs_transformed')

# print("DataFrame written to Hive table successfully!")

# COMMAND ----------

df1.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# COMMAND ----------

# "customer_name",
# "CR_Rating",
# "YRS_In_BUS",
# "TNW_in_MEUR",
# "profits_perc_TNW",
# "Positive_WC",
# "TNW_to_T-Exposure",
# "Date_of_Est.
# "financing_currency",
# "fleet_size",
# "pmt_discipline",
# "total_exposure",
# "revenue",
# "EBIT",
# "depreciation",
# "net_profit",
# "fixed_assets",
# "intangible_assets",
# "current_assets",
# "tangible_net_worth",
# "long_term_liab",
# "long_term_credit",
# "short_term_liab",
# "short_term_credit",
# "off_balance_liab",
# "CR_Rating_score",
# "target_variable",
# "short_term_leverage",

# COMMAND ----------

df1.printSchema()

# COMMAND ----------

# from pyspark.sql.functions import when, col 

# columns = ["profitability_ratio", "asset_to_equity_ratio", "current_ratio", "revenue_growth","net_profit_growth", "debt_service_capacity", "PPE_to_total_Assets_Ratio"]

# for c in columns:
#     df1 = df1.withColumn(c, when(col(c).isNull(), 0).otherwise(col(c)))

# df.show
# display(df1)

# COMMAND ----------

# spark.sql('DROP TABLE IF EXISTS hive_metastore.default.crs_transformed')
# df1.write.mode('overwrite').saveAsTable('hive_metastore.default.crs_transformed')

# COMMAND ----------

# corr_matrix = df1.select(columns).toPandas().corr()
# print(corr_matrix)

# COMMAND ----------


