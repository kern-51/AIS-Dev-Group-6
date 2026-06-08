#!/bin/bash

# Exit immediately if any command returns a non-zero status
set -e

echo "--- Starting ML Pipeline ---"

# 1. Data Ingestion
echo "Step 1: Running Ingestion..."
python src/ingest.py

# 2. Data Cleaning
echo "Step 2: Running Cleaning..."
python src/clean.py

# 3. Feature Engineering
echo "Step 3: Running Feature Engineering..."
python src/feature_engineering.py

# 4. Model Training
echo "Step 4: Running Random Forest Model Training..."
python src/random_forest.py

# 5. Model Training
echo "Step 5: Running Logistic Regression Model Training..."
python src/logistic_regression.py

# 6. Model Training
echo "Step 6: Running Isolated Forest Model Training..."
python src/isolated_forest.py

echo "Step 7: Running main.py"
python src/main.py

echo "--- Pipeline Finished Successfully ---"