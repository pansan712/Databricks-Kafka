from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, udf, explode, array, lit
from pyspark.sql.types import DoubleType, ArrayType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import mlflow
import mlflow.spark

# Initialize Spark session
spark = SparkSession.builder.appName("BankingFraudDetection").getOrCreate()

# Load transaction data
transactions_df = spark.read.parquet("/path/to/transactions_data")

# Load customer data
customers_df = spark.read.parquet("/path/to/customers_data")

# Join transactions with customer data
joined_df = transactions_df.join(customers_df, "customer_id")

# Feature engineering
def calculate_transaction_frequency(df):
    return df.groupBy("customer_id").agg({"transaction_id": "count"}).withColumnRenamed("count(transaction_id)", "transaction_frequency")

transaction_frequency_df = calculate_transaction_frequency(transactions_df)

# Join with the main dataframe
joined_df = joined_df.join(transaction_frequency_df, "customer_id")

# Create features for the model
feature_columns = ["amount", "transaction_frequency", "customer_age", "account_age_days"]
label_column = "is_fraud"

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Split the data
train_data, test_data = joined_df.randomSplit([0.8, 0.2], seed=42)

# Define the model
rf = RandomForestClassifier(labelCol=label_column, featuresCol="features", numTrees=100)

# Create a pipeline
pipeline = Pipeline(stages=[assembler, rf])

# Enable MLflow tracking
mlflow.spark.autolog()

with mlflow.start_run(run_name="Fraud Detection Model"):
    # Train the model
    model = pipeline.fit(train_data)
    
    # Make predictions
    predictions = model.transform(test_data)
    
    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol=label_column)
    auc = evaluator.evaluate(predictions)
    print(f"AUC: {auc}")

    # Log custom metrics
    mlflow.log_metric("AUC", auc)

# Save the model
model.write().overwrite().save("/path/to/fraud_detection_model")

# Function to detect anomalies in real-time
@udf(returnType=ArrayType(DoubleType()))
def detect_anomalies(amount, transaction_frequency, customer_age, account_age_days):
    # Load the saved model
    loaded_model = Pipeline.load("/path/to/fraud_detection_model")
    
    # Create a DataFrame with the input data
    input_data = spark.createDataFrame([(amount, transaction_frequency, customer_age, account_age_days)], 
                                       ["amount", "transaction_frequency", "customer_age", "account_age_days"])
    
    # Make prediction
    result = loaded_model.transform(input_data)
    
    # Return probability of fraud
    return result.select("probability").collect()[0][0]

# Apply the UDF to incoming transactions
streaming_transactions = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "transactions") \
    .load()

# Process streaming data
processed_stream = streaming_transactions \
    .selectExpr("CAST(value AS STRING)") \
    .select(
        explode(array(
            lit("amount"), 
            lit("transaction_frequency"), 
            lit("customer_age"), 
            lit("account_age_days")
        )).alias("feature"),
        col("value")
    ) \
    .groupBy("value") \
    .pivot("feature") \
    .agg({"value": "first"}) \
    .select(
        col("amount").cast("double"),
        col("transaction_frequency").cast("double"),
        col("customer_age").cast("double"),
        col("account_age_days").cast("double")
    )

# Apply anomaly detection
anomaly_stream = processed_stream \
    .withColumn("anomaly_score", detect_anomalies(
        col("amount"), 
        col("transaction_frequency"), 
        col("customer_age"), 
        col("account_age_days")
    ))

# Write results to Delta Lake
query = anomaly_stream \
    .writeStream \
    .outputMode("append") \
    .format("delta") \
    .option("checkpointLocation", "/path/to/checkpoint") \
    .start("/path/to/anomalies_table")

query.awaitTermination()
