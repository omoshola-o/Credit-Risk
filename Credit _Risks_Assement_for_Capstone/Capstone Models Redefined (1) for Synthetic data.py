# Databricks notebook source
# MAGIC %pip install sdv
# MAGIC %pip install copulas
# MAGIC %pip install Faker
# MAGIC %pip install sdmetrics
# MAGIC %pip install fitter

# COMMAND ----------

import pyspark
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder \
  .master('local[1]') \
  .appName('cra_mvp') \
  .getOrCreate()

df = spark.read.table('hive_metastore.default.crs_transformed')
df.printSchema()
display(df)

# COMMAND ----------

df = df.toPandas()

# COMMAND ----------

data =  df.columns

# COMMAND ----------

data_to_synthesize = df[['TNW_in_MEUR', 'profits_perc_TNW', 'Positive_WC', 'TNW_to_T-Exposure', 'fleet_size', 'total_exposure', 'revenue', 'EBIT', 'depreciation', 'net_profit', 'fixed_assets', 'intangible_assets', 'current_assets', 'tangible_net_worth', 'long_term_liab', 'long_term_credit', 'short_term_liab', 'CR_Rating_score', 'pmt_discipline_score', 'debt_equity_ratio', 'debt_asset_ratio', 'current_ratio', 'return_on_assets', 'target_variable']]

# COMMAND ----------

display(data_to_synthesize)

# COMMAND ----------

import sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.constraints import FixedCombinations
from sdv.constraints import Unique
from sdv.constraints import Unique
from sdv.constraints import FixedCombinations


# COMMAND ----------

from sdv.metadata import SingleTableMetadata
metadata = SingleTableMetadata()
df = data_to_synthesize
metadata.detect_from_dataframe(df)

# COMMAND ----------

metadata.visualize

# COMMAND ----------

python_dict = metadata.to_dict()

# COMMAND ----------

metadata.validate()

# COMMAND ----------

synthesizer = GaussianCopulaSynthesizer(metadata,enforce_min_max_values=True,enforce_rounding=True,
    numerical_distributions={'TNW_in_MEUR': 'beta',
                            'profits_perc_TNW': 'beta',
                            'Positive_WC': 'beta',
                            'TNW_to_T-Exposure': 'beta',
                            'fleet_size': 'beta',
                            'total_exposure': 'beta',
                            'revenue': 'beta',
                            'EBIT': 'beta', 
                            'depreciation': 'beta', 
                            'net_profit': 'beta',
                            'fixed_assets': 'beta', 
                            'intangible_assets': 'beta', 
                            'current_assets': 'beta',
                            'tangible_net_worth': 'beta,
                            'long_term_liab': 'beta', 
                            'long_term_credit': 'beta', 
                            'short_term_liab': 'beta', 
                            'CR_Rating_score': 'beta', 
                            'pmt_discipline_score': 'beta', 
                            'debt_equity_ratio': 'beta', 
                            'debt_asset_ratio': 'beta', 
                            'current_ratio': 'beta', 
                            'return_on_assets': 'beta', 
                            'target_variable': 'beta'},
    default_distribution='truncnorm')
    #default_distribution='norm'

# COMMAND ----------

synthesizer.fit(df)

# COMMAND ----------

#synthetic_data = synthesizer.sample(num_rows=len(data)*10)
synthetic_data = synthesizer.sample(num_rows=len(data))
#synthetic_data = synthesizer.sample(num_rows=10000)
display(synthetic_data)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count plot for 'Exited' column
sns.countplot(x='target_variable', data=df, hue='target_variable', palette='Blues', ax=axes[1])
axes[1].set_title('Count Plot of Exited')

# Pie chart for 'Exited' column
status_counts = df['target_variable'].value_counts()
axes[0].pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Blues'))
axes[0].set_title('Distribution of Exited')



plt.tight_layout()
plt.show()
