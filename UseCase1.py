from datetime import datetime
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException
import pandas as pd
import logging
import json

logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from JSON file
with open('D:/ByteSizeProject/usecase1/s3_bucket/Code/config.json') as config_file:
    config = json.load(config_file)

spark = SparkSession.builder \
    .appName("UseCase Data Flow Pipeline") \
    .getOrCreate()

def read_file_to_df(file_path):
    """Read a CSV file into a Spark DataFrame."""
    return spark.read.csv(file_path, header=True, inferSchema=True)

def read_counter_file(counter_file_path):
    """Read the expected row count from the counter file."""
    counter_df = read_file_to_df(counter_file_path)
    if counter_df.count() == 0:
        raise ValueError(f"Counter file is empty: {counter_file_path}")
    return counter_df.collect()[0][0]

def read_header_file(header_file_path):
    """Read the expected headers from the header file."""
    header_df = read_file_to_df(header_file_path)
    return header_df.columns  # Return the column names as a list

def validate_data(spark_df, expected_count, expected_headers):
    """Perform row count, header, and null value validations on the DataFrame."""
    # Check row count
    actual_count = spark_df.count()
    if actual_count != expected_count:
        raise ValueError(f"Row count mismatch: expected {expected_count}, found {actual_count}.")

    # Check headers
    actual_headers = spark_df.columns
    if sorted(actual_headers) != sorted(expected_headers):
        raise ValueError("Header mismatch. Expected headers do not match actual headers.")

    # Check for NULL values in all columns and fill with 'NA'
    null_columns = [header for header in actual_headers if spark_df.filter(F.col(header).isNull()).count() > 0]
    if null_columns:
        spark_df = spark_df.fillna('NA')
        logger.info(f"Filled NULL values in columns: {null_columns}")

def copy_file_to_preprocess(file_path, preprocess_folder):
    """Copy the file to the preprocess folder."""
    file_name = os.path.basename(file_path)
    destination_file = os.path.join(preprocess_folder, file_name)
    shutil.copy(file_path, destination_file)
    return destination_file

def convert_xls_to_csv(xls_file, csv_file):
    """Convert XLS file to CSV with ',' delimiter."""
    df = pd.read_excel(xls_file)
    df.to_csv(csv_file, sep=',', index=False)

def inbound_to_preprocess(inbound_folder, preprocess_folder):
    """Copy files from inbound to preprocess folder."""
    logger.info(f"Processing files from {inbound_folder} to {preprocess_folder}")
    for company_folder in os.listdir(inbound_folder):
        company_folder_path = os.path.join(inbound_folder, company_folder)
        if os.path.isdir(company_folder_path):
            for file_name in os.listdir(company_folder_path):
                file_path = os.path.join(company_folder_path, file_name)
                file_extension = os.path.splitext(file_name)[1].lower()
                
                try:
                    # Process .xls files
                    if file_extension == '.xls':
                        csv_file_path = os.path.join(preprocess_folder, f"{file_name.replace('.xls', '.csv')}")
                        convert_xls_to_csv(file_path, csv_file_path)
                        logger.info(f"Converted {file_name} to CSV and wrote to {csv_file_path}")


                    # Process .csv files
                    elif file_extension == '.csv':
                        csv_file_path = copy_file_to_preprocess(file_path, preprocess_folder)
                        logger.info(f"Copied {file_name} to {csv_file_path}")

                    else:
                        logger.warning(f"Unsupported file format: {file_name}")

                except Exception as e:
                    logger.error(f"Error processing file {file_name}: {e}", exc_info=True)

def preprocess_to_landing(preprocess_folder, landing_folder, counter_file_path, header_file_path):
    """Process files from the preprocess folder to the landing folder."""
    expected_count = read_counter_file(counter_file_path)
    expected_headers = read_header_file(header_file_path)

    for file_name in os.listdir(preprocess_folder):
        file_path = os.path.join(preprocess_folder, file_name)

        try:
            # Load CSV file into Spark DataFrame
            spark_df = read_file_to_df(file_path)

            # Perform validations
            validate_data(spark_df, expected_count, expected_headers)

            # Copy the file from preprocess folder to landing folder
            destination_file_path = os.path.join(landing_folder, file_name)
            shutil.copy(file_path, destination_file_path)
            logger.info(f"Copied {file_name} to {landing_folder}")

        except ValueError as ve:
            logger.error(f"Validation error for file {file_name}: {ve}")
            # Skip copying to landing if validation fails
        except Exception as e:
            logger.error(f"Error processing file to landing: {e}")

def landing_to_standardized(landing_folder, standardized_folder):
    """Process files from the landing folder to the standardized folder."""
    for file_name in os.listdir(landing_folder):
        file_path = os.path.join(landing_folder, file_name)
        try:
            df = read_file_to_df(file_path)
            # Separate date column into multiple columns
            df = df.withColumnRenamed("script_name", "company_name") \
                .withColumn("date", F.to_date(F.col("Date"))) \
                .withColumn("year", F.year(F.col("date"))) \
                .withColumn("month", F.month(F.col("date"))) \
                .withColumn("quarter", F.quarter(F.col("date")))

            # Write to standardized folder
            df.write.partitionBy("company_name").parquet(standardized_folder, mode='overwrite')
            logger.info(f"Loaded data to standardized {standardized_folder}")

        except Exception as e:
            logger.error(f"Error processing file to standardized: {e}")

def summarize_data(standardized_folder, summary_folder):
    """Summarize data for every company and create CSV summaries."""
    df = spark.read.parquet(standardized_folder)

    # Summarize data for each company
    summary_df = df.groupBy("company_name") \
        .agg(
            F.count("*").alias("total_rows"),
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date")
        )

    # Write summary to CSV with a constant name
    summary_file_path = os.path.join(summary_folder, "company_summary.csv")
    summary_df.write.csv(summary_file_path, header=True, mode='overwrite')

    logger.info(f"Summary CSV created at: {summary_file_path}")

def move_summary_to_outbound(summary_folder, outbound_folder, archive_folder):
    summary_file = os.path.join(summary_folder, "company_summary.csv")
    destination_file = os.path.join(outbound_folder, "company_summary.csv")    

    if os.path.exists(destination_file):
        # If the file exists in the outbound folder, move it to the archive folder with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        archived_file = os.path.join(archive_folder, f"company_summary_{timestamp}.csv")
        shutil.move(destination_file, archived_file)
        logger.info(f"Existing file moved to archive: {archived_file}")

    # Move the new summary file to the outbound folder
    shutil.move(summary_file, destination_file)
    logger.info(f"Moved summary file to {destination_file}")
    


if __name__ == "__main__":
    # Define the paths
    inbound_folder = config['inbound_folder']
    preprocess_folder = config['preprocess_folder']
    landing_folder = config['landing_folder']
    standardized_folder = config['standardized_folder']
    counter_file_path = config['counter_file_path']
    header_file_path = config['header_file_path']
    summary_folder = config['summary_folder']
    outbound_folder = config['outbound_folder']
    archive_folder = config['archive_folder']

    # Ensure folders exist
    os.makedirs(preprocess_folder, exist_ok=True)
    os.makedirs(landing_folder, exist_ok=True)
    os.makedirs(standardized_folder, exist_ok=True)
    os.makedirs(summary_folder, exist_ok=True)
    os.makedirs(outbound_folder, exist_ok=True)
    os.makedirs(archive_folder, exist_ok=True)

    # Run the data pipeline 
    inbound_to_preprocess(inbound_folder, preprocess_folder)
    preprocess_to_landing(preprocess_folder, landing_folder, counter_file_path, header_file_path)
    landing_to_standardized(landing_folder, standardized_folder)
    summarize_data(standardized_folder, summary_folder)
    move_summary_to_outbound(summary_folder, outbound_folder, archive_folder)

# Stop Spark session
spark.stop()
