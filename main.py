from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import pandas as pd
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

try:
    logger.info("Initializing Spark session...")
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Churn Prediction API") \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "1g") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .getOrCreate()
    logger.info("Spark session initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Spark session: {str(e)}")
    raise

# Define schema for input data
schema = StructType([
    StructField("tenure", FloatType(), True),
    StructField("MonthlyCharges", FloatType(), True),
    StructField("TotalCharges", FloatType(), True),
    StructField("gender", StringType(), True),
    StructField("SeniorCitizen", StringType(), True),
    StructField("Partner", StringType(), True),
    StructField("Dependents", StringType(), True),
    StructField("PhoneService", StringType(), True),
    StructField("MultipleLines", StringType(), True),
    StructField("InternetService", StringType(), True),
    StructField("OnlineSecurity", StringType(), True),
    StructField("OnlineBackup", StringType(), True),
    StructField("DeviceProtection", StringType(), True),
    StructField("TechSupport", StringType(), True),
    StructField("StreamingTV", StringType(), True),
    StructField("StreamingMovies", StringType(), True),
    StructField("Contract", StringType(), True),
    StructField("PaperlessBilling", StringType(), True),
    StructField("PaymentMethod", StringType(), True)
])

# Load the RandomForest model
model_path = "rf_spark_model"
try:
    logger.info(f"Attempting to load model from {model_path}")
    # Verify model directory exists and has required files
    if not os.path.exists(model_path):
        raise Exception(f"Model directory {model_path} does not exist")
    
    metadata_path = os.path.join(model_path, "metadata")
    if not os.path.exists(metadata_path):
        raise Exception(f"Model metadata directory does not exist at {metadata_path}")
    
    # Verify metadata file exists
    metadata_file = os.path.join(metadata_path, "part-00000")
    if not os.path.exists(metadata_file):
        raise Exception(f"Model metadata file does not exist at {metadata_file}")
    
    logger.info("Model directory structure verified, loading model...")
    model = RandomForestClassificationModel.load(model_path)
    logger.info(f"Successfully loaded model from {model_path}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Directory contents: {os.listdir('.')}")
    if os.path.exists(model_path):
        logger.error(f"Model directory contents: {os.listdir(model_path)}")
        if os.path.exists(metadata_path):
            logger.error(f"Metadata directory contents: {os.listdir(metadata_path)}")
    raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

# Pydantic model for request validation
class CustomerData(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: str
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        spark_df = spark.createDataFrame(input_data, schema=schema)

        # Define feature columns
        categorical_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
            "Contract", "PaperlessBilling", "PaymentMethod"
        ]
        numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

        # Preprocessing stages
        stages = []
        for col in categorical_columns:
            indexer = StringIndexer(inputCol=col, outputCol=col + "_index")
            encoder = OneHotEncoder(inputCols=[col + "_index"], outputCols=[col + "_encoded"])
            stages += [indexer, encoder]

        assembler_inputs = [col + "_encoded" for col in categorical_columns] + numerical_columns
        assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
        stages.append(assembler)

        # Apply preprocessing
        from pyspark.ml import Pipeline
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(spark_df)
        transformed_df = pipeline_model.transform(spark_df)

        # Make prediction
        prediction = model.transform(transformed_df)
        result = prediction.select("prediction").collect()[0]["prediction"]
        
        return {"prediction": int(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application...")
    try:
        # Run the application with explicit host and port
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False,
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise