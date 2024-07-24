from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SupplyChainOptimization") \
    .getOrCreate()

# Read data from Kafka
df = (spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "<server:ip>")
      .option("subscribe", "<input_topic>")
      .option("startingOffsets", "latest")
      .load())

# Convert binary values to strings
df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")

# Define the schema of the JSON data
schema = StructType([
    StructField("item_id", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("timestamp", StringType(), True)
])

# Parse the JSON data
parsed_df = df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Example transformation: Aggregate total quantity by item_id
agg_df = parsed_df.groupBy("item_id").sum("quantity").alias("total_quantity")

# Write the aggregated data back to Kafka
query = (agg_df
         .selectExpr("CAST(item_id AS STRING) AS key", "CAST(total_quantity AS STRING) AS value")
         .writeStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "<server:ip>")
         .option("topic", "<output_topic>")
         .option("checkpointLocation", "/path/to/checkpoint/dir")
         .outputMode("complete")
         .start())

query.awaitTermination()
