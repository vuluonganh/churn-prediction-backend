#!/usr/bin/env bash
apt-get update
apt-get install -y openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH

# Now run your app
uvicorn main:app --host 0.0.0.0 --port 10000

# Install dependencies
pip install -r requirements.txt

