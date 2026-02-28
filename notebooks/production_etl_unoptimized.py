# Databricks notebook source
# MAGIC %md
# MAGIC # Production ETL Job - Customer Data Pipeline
# MAGIC 
# MAGIC **Purpose:** Process customer transactions and enrich with reference data.
# MAGIC **Schedule:** Daily at 2:00 AM
# MAGIC **Owner:** Data Engineering Team

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Imports

# COMMAND ----------

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType
from pyspark.sql import Row
import time

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Parameters and Widgets

# COMMAND ----------

# Create notebook widgets for runtime configuration
dbutils.widgets.text("source_path", "/mnt/raw/transactions/", "Source Data Path")
dbutils.widgets.text("reference_path", "/mnt/raw/lookup_tables/", "Reference Data Path")
dbutils.widgets.text("output_path", "/mnt/bronze/processed_transactions/", "Output Delta Path")
dbutils.widgets.text("partition_date", "", "Partition Date (YYYY-MM-DD)")

# Retrieve widget values
SOURCE_PATH = dbutils.widgets.get("source_path")
REFERENCE_PATH = dbutils.widgets.get("reference_path")
OUTPUT_PATH = dbutils.widgets.get("output_path")
PARTITION_DATE = dbutils.widgets.get("partition_date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Read Source Data - Schema Inference
# MAGIC 
# MAGIC **Note:** Using inferSchema for flexibility with evolving CSV structure.

# COMMAND ----------

# ANTI-PATTERN: Schema Inference
# Reading large CSV with inferSchema=true forces Spark to scan the entire file
# to infer column types. For production, always provide explicit DDL schema.
transactions_df = (
    spark.read
    .format("csv")
    .option("header", "true")
    .option("inferSchema", "true")  # Expensive: full file scan before any processing
    .option("delimiter", ",")
    .load(f"{SOURCE_PATH}/*.csv")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Read Reference Data - Lack of Predicate Pushdown
# MAGIC 
# MAGIC **Note:** Load full dimension table before filtering for required keys.

# COMMAND ----------

# ANTI-PATTERN: Lack of Pruning / Predicate Pushdown
# Read the ENTIRE lookup table into memory before applying any filter.
# Should instead use: spark.table("lookup_table").filter(F.col("region") == "active")
# or pass partition predicates at read time.
lookup_table_full = spark.read.format("delta").load(f"{REFERENCE_PATH}/region_lookup")
# Now filter AFTER loading everything - no predicate pushdown benefit
lookup_df = lookup_table_full.filter(F.col("is_active") == True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Row-by-Row Processing via Python UDF
# MAGIC 
# MAGIC **Note:** Custom business logic for standardizing customer identifiers.

# COMMAND ----------

# ANTI-PATTERN: Row-by-Row Processing with Python UDF
# Python UDFs execute row-by-row on the driver/JVM boundary with serialization overhead.
# Use built-in Spark SQL functions instead: F.upper(F.trim(F.col("customer_id")))

@F.udf(StringType())
def standardize_customer_id(customer_id: str) -> str:
    """Standardize customer ID format for downstream systems."""
    if customer_id is None:
        return None
    return customer_id.strip().upper()

# Apply UDF to every row - serialization/deserialization for each row
transactions_df = transactions_df.withColumn(
    "standardized_customer_id",
    standardize_customer_id(F.col("customer_id"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: The Collect Trap
# MAGIC 
# MAGIC **Note:** Aggregating statistics for validation - requires driver access.

# COMMAND ----------

# ANTI-PATTERN: The Collect Trap
# .collect() pulls ALL data to the driver - memory exhaustion risk for large datasets.
# Processing in Python for-loop defeats distributed computing. Should use Spark
# aggregations and actions instead.
collected_data = transactions_df.select("standardized_customer_id", "amount", "region_id").collect()

# Process each row in a Python for-loop on the driver - single-threaded, no parallelism
processed_rows = []
for row in collected_data:
    customer_id = row["standardized_customer_id"]
    amount = row["amount"]
    region_id = row["region_id"]
    # Simulate "business logic" that could be done with Spark
    adjusted_amount = amount * 1.05 if region_id in [1, 2, 3] else amount
    processed_rows.append(Row(
        customer_id=customer_id,
        amount=amount,
        region_id=region_id,
        adjusted_amount=adjusted_amount
    ))

# Parallelize back to RDD then to DataFrame - expensive round-trip
transactions_df = spark.sparkContext.parallelize(processed_rows).toDF()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Inefficient Joins - No Broadcast, Data Skew
# MAGIC 
# MAGIC **Note:** Enrich transactions with region metadata. Lookup table is small but not broadcast.

# COMMAND ----------

# Create skewed data scenario: most transactions have region_id = 1
# ANTI-PATTERN: Small table joined without broadcast hint
# lookup_df has ~10 rows but Spark may choose Sort-Merge Join, shuffling both sides.
# Should use: broadcast(lookup_df) or F.broadcast(lookup_df)

# ANTI-PATTERN: Join key has severe data skew - region_id=1 dominates
# Causes hotspot on single partition during shuffle
enriched_df = transactions_df.join(
    lookup_df,  # Small table - should use broadcast(lookup_df)
    transactions_df.region_id == lookup_df.region_id,
    "left"
).select(
    transactions_df["*"],
    lookup_df.region_name,
    lookup_df.region_tier
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Excessive Shuffling via Repartition
# MAGIC 
# MAGIC **Note:** Preparing data for write - ensure optimal partition distribution.

# COMMAND ----------

# ANTI-PATTERN: Excessive Shuffling
# Multiple .repartition() calls each trigger a full shuffle. Use .coalesce() when
# reducing partitions (no shuffle), or single repartition by key when needed.
enriched_df = enriched_df.repartition(200)  # Shuffle #1
enriched_df = enriched_df.repartition(50, "region_id")  # Shuffle #2
enriched_df = enriched_df.repartition(100)  # Shuffle #3 - completely undoes previous partitioning
# Should use: single .repartition(n, "partition_column") or .coalesce(n) when reducing

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Write to Delta Lake

# COMMAND ----------

# Write final output to Delta table
# Partitioning by date is reasonable, but data has already been shuffled excessively
(
    enriched_df
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .partitionBy("region_id")
    .save(OUTPUT_PATH)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Job Completion Logging

# COMMAND ----------

print(f"ETL completed successfully. Output written to {OUTPUT_PATH}")
dbutils.notebook.exit("SUCCESS")
