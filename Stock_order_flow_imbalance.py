from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize Spark session
spark = SparkSession.builder \
    .appName("RealTimeStockOrderFlowAnalysis") \
    .getOrCreate()

# Define schema for incoming stock order data
schema = StructType([
    StructField("timestamp", TimestampType(), True),
    StructField("symbol", StringType(), True),
    StructField("order_type", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", DoubleType(), True)
])

# Read streaming data from Kafka
df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "your_kafka_bootstrap_servers") \
    .option("subscribe", "stock_orders") \
    .load()

# Parse JSON data from Kafka
parsed_df = df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

# Calculate order flow imbalance
window_duration = "1 minute"
slide_duration = "30 seconds"

order_flow_imbalance = parsed_df \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy(
        window("timestamp", window_duration, slide_duration),
        "symbol"
    ) \
    .agg(
        sum(when(col("order_type") == "BUY", col("quantity")).otherwise(0)).alias("buy_volume"),
        sum(when(col("order_type") == "SELL", col("quantity")).otherwise(0)).alias("sell_volume")
    ) \
    .withColumn("order_flow_imbalance", col("buy_volume") - col("sell_volume"))

# Write results to Delta Lake
query = order_flow_imbalance \
    .writeStream \
    .outputMode("append") \
    .format("delta") \
    .option("checkpointLocation", "/path/to/checkpoint") \
    .start("/path/to/order_flow_imbalance_table")

# Wait for the streaming query to terminate
query.awaitTermination()
