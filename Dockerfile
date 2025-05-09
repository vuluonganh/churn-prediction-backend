# Use an official Python runtime as the base image
FROM python:3.9-slim

# Update package sources and install Java
RUN apt-get update && apt-get install -y \
    curl \
    && echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list \
    && echo "deb http://deb.debian.org/debian-security bullseye-security main" >> /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64
ENV PATH $JAVA_HOME/bin:$PATH

# Set working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory with proper permissions
RUN mkdir -p /app/rf_spark_model && \
    chmod -R 755 /app/rf_spark_model

# Copy the model files
COPY rf_spark_model/ /app/rf_spark_model/

# Set proper permissions and verify
RUN chmod -R 755 /app/rf_spark_model && \
    chown -R root:root /app/rf_spark_model && \
    ls -la /app/rf_spark_model/ && \
    ls -la /app/rf_spark_model/metadata/

# Copy the rest of the application
COPY . .

# Set environment variables for Spark
ENV SPARK_LOCAL_IP=0.0.0.0
ENV SPARK_PUBLIC_DNS=localhost
ENV PYTHONUNBUFFERED=1
ENV SPARK_HOME=/usr/local/lib/python3.9/site-packages/pyspark
ENV HADOOP_CONF_DIR=/usr/local/lib/python3.9/site-packages/pyspark/conf

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["python", "-u", "main.py"]