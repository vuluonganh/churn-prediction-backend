from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassificationModel
from typing import Optional
from pyngrok import ngrok
import uvicorn

# Set ngrok authtoken
ngrok.set_auth_token("2wmGl8hgRgaDaHXZ1dNO0kOjdma_3foKNMhhQBM3veuo8tf5s")

# Create Spark session
spark = SparkSession.builder.master("local[*]").getOrCreate()

# Load the RandomForest model (since you have rf_spark_model)
model = RandomForestClassificationModel.load("rf_spark_model")

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
    Dependents: str
    tenure: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to Spark DataFrame
        input_dict = data.model_dump()
        input_row = Row(**input_dict)
        input_df = spark.createDataFrame([input_row])

        # Preprocess categorical features since model is RandomForestClassificationModel
        categorical_cols = [
            "Dependents", "InternetService", "OnlineSecurity", "TechSupport",
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"
        ]
        numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

        # Step 1: Use StringIndexer to convert categorical columns to numerical indices
        indexers = [
            StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            for col in categorical_cols
        ]

        # Step 2: One-hot encode the indexed columns
        encoders = [
            OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_encoded")
            for col in categorical_cols
        ]

        # Step 3: Assemble all features into a single vector
        feature_cols = numerical_cols + [f"{col}_encoded" for col in categorical_cols]
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )

        # Create a pipeline for preprocessing
        from pyspark.ml import Pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler])
        input_df = pipeline.fit(input_df).transform(input_df)

        # Make prediction
        prediction = model.transform(input_df)

        # Extract prediction result
        if "prediction" in prediction.columns:
            pred_value = prediction.select("prediction").collect()[0][0]
            is_churn = bool(pred_value == 1.0)

            prob_value = None
            if "probability" in prediction.columns:
                prob_array = prediction.select("probability").collect()[0][0]
                prob_value = float(prob_array[1])

            return {
                "result": "Yes" if is_churn else "No",
                "probability": prob_value if prob_value is not None else None
            }
        else:
            raise HTTPException(status_code=500, detail="Prediction column not found in model output")

    except Exception as e:
        import traceback
        print("Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(8000)
    print("ngrok tunnel:", public_url)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)