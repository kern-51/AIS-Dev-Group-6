import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def isolated_forest(
    df, contamination=0.05, random_state=42
):
    """Builds an Isolation Forest pipeline, fits it to the dataframe,

    and returns a copy of the dataframe with anomaly flags and scores,
    along with the trained pipeline object.
    """
    # Create a copy to avoid altering original data
    df_out = df.copy()
    # numeric columns
    numeric_cols = [ 'Temperature',
    'Humidity',
    'CO2_InfraredSensor',
    'CO2_ElectroChemicalSensor',
    'MetalOxideSensor_Unit1',
    'MetalOxideSensor_Unit2',
    'MetalOxideSensor_Unit3',
    'MetalOxideSensor_Unit4'
    ]
    # differentiate types of categorical data
    ordinal_cols = [
    'Time of Day',
    'Ambient Light Level',
    'Activity Level'
    ]
    nominal_cols = [
    'HVAC Operation Mode',
    ]
    
    # Setup the preprocessing layers
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("ord", "passthrough", ordinal_cols),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    ]
    )
    # Define the full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "anomaly_detector",
                IsolationForest(
                    n_estimators=100,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Fit the pipeline and extract predictions
    pipeline.fit(df_out)

    # 1 = Normal, -1 = Anomaly
    df_out["Anomaly_Flag"] = pipeline.predict(df_out)

    # Continuous score (lower/negative = more anomalous)
    df_out["Anomaly_Score"] = pipeline.decision_function(df_out)

    return df_out, pipeline
