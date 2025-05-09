FROM python:3.8-slim

# Install OpenJDK
RUN apt-get update && apt-get install -y openjdk-11-jdk && apt-get clean

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and model
COPY app.py .
COPY rf_spark_model /rf_spark_model

# Set the working directory
WORKDIR /

# Run the app
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]