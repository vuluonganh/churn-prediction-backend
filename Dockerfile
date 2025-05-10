# Use an official Python runtime as the base image
FROM python:3.9-slim

# Install Java and other dependencies
RUN apt-get update && apt-get install -y \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Find and set the correct Java home
RUN java_home=$(dirname $(dirname $(readlink -f $(which java)))) && \
    echo "Found Java home at: $java_home" && \
    echo "export JAVA_HOME=$java_home" >> /etc/profile && \
    echo "export PATH=$JAVA_HOME/bin:$PATH" >> /etc/profile

# Set environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV SPARK_LOCAL_IP=0.0.0.0

# Verify Java installation
RUN java -version && \
    echo "JAVA_HOME is set to: $JAVA_HOME" && \
    ls -l $JAVA_HOME/bin/java

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application code
COPY svm_spark_model/ ./svm_spark_model/
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application with proper signal handling
CMD ["python", "main.py"]