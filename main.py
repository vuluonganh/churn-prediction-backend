from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LinearSVCModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
import logging
import os
import sys
import atexit
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create Spark session with proper configuration
spark = SparkSession.builder \
    .appName("Churn Prediction API") \
    .master("local[1]") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "1g") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.fs.defaultFS", "file:///") \
    .config("spark.driver.extraJavaOptions", "-Dfile.encoding=UTF-8") \
    .config("spark.executor.extraJavaOptions", "-Dfile.encoding=UTF-8") \
    .getOrCreate()

# Register cleanup function
def cleanup():
    logger.info("Cleaning up Spark session...")
    if spark:
        spark.stop()
        logger.info("Spark session stopped")

atexit.register(cleanup)

# Load the SVM model
model_path = "model/svm_spark_model"
try:
    logger.info(f"Attempting to load model from {model_path}")
    if not os.path.exists(model_path):
        raise Exception(f"Model directory {model_path} does not exist")
    
    metadata_path = os.path.join(model_path, "metadata")
    if not os.path.exists(metadata_path):
        raise Exception(f"Model metadata directory does not exist at {metadata_path}")
    
    metadata_file = os.path.join(metadata_path, "part-00000")
    if not os.path.exists(metadata_file):
        raise Exception(f"Model metadata file does not exist at {metadata_file}")
    
    logger.info("Model directory structure verified, loading model...")
    model = LinearSVCModel.load(model_path)
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

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data structure
class InputData(BaseModel):
    # Numerical features first
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # Categorical features
    Dependents: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to Spark DataFrame
        input_dict = data.model_dump()
        input_row = Row(**input_dict)
        input_df = spark.createDataFrame([input_row])

        # Define features in exact order as trained model
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = [
            "Dependents", "InternetService", "OnlineSecurity", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
        ]

        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            for col in categorical_cols
        ]

        onehotencoder = [
            OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded", handleInvalid="keep")
            for col in categorical_cols
        ]

        feature_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="num_features"
        )

        scaler = StandardScaler(
            inputCol="num_features",
            outputCol="features",
        )

        pipeline = Pipeline(stages=indexers + onehotencoder +[assembler] + [scaler])
        input_df = pipeline.fit(input_df).transform(input_df)

        # Make prediction
        prediction = model.transform(input_df)

        # Extract prediction result and raw prediction score
        if "prediction" in prediction.columns and "rawPrediction" in prediction.columns:
            pred_row = prediction.select("prediction", "rawPrediction").collect()[0]
            pred_value = pred_row[0]
            raw_pred = pred_row[1][1]  # Get the score for class 1 (churn)
            
            # Log the raw prediction values for debugging
            logger.info(f"Raw prediction values: {pred_row[1]}")
            logger.info(f"Selected raw prediction for class 1: {raw_pred}")
            
            # Scale the raw prediction to a more reasonable range
            # Using a scaling factor to bring the values into a reasonable range
            scaled_pred = raw_pred / 100.0  # Adjust this scaling factor based on your model's output range
            
            # Convert scaled prediction to probability using sigmoid
            probability = 1 / (1 + np.exp(-scaled_pred))
            logger.info(f"Scaled prediction: {scaled_pred}")
            logger.info(f"Calculated probability: {probability}")
            
            is_churn = bool(pred_value == 1.0)
            logger.info(f"Final prediction: {is_churn}")

            return {
                "result": "Yes" if is_churn else "No",
                "probability": float(probability),
                "raw_prediction": float(raw_pred),
                "scaled_prediction": float(scaled_pred)
            }
        else:
            raise HTTPException(status_code=500, detail="Prediction or rawPrediction column not found in model output")

    except Exception as e:
        import traceback
        print("Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Test Spark session
        spark.sql("SELECT 1").collect()
        return {"status": "healthy", "spark": "connected", "model": "loaded"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting application...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        cleanup()
        raise