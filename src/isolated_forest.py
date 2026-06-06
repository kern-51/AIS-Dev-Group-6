import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def isolated_forest(
    df, numeric_cols, categorical_cols, contamination=0.05, random_state=42
):
    """Builds an Isolation Forest pipeline, fits it to the dataframe,

    and returns a copy of the dataframe with anomaly flags and scores,
    along with the trained pipeline object.
    """
    # 1. Create a copy to avoid altering original data
    df_out = df.copy()

    # 2. Setup the preprocessing layers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", "passthrough", categorical_cols),
        ]
    )

    # 3. Define the full pipeline
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

    # 4. Fit the pipeline and extract predictions
    pipeline.fit(df_out)

    # 1 = Normal, -1 = Anomaly
    df_out["Anomaly_Flag"] = pipeline.predict(df_out)

    # Continuous score (lower/negative = more anomalous)
    df_out["Anomaly_Score"] = pipeline.decision_function(df_out)

    return df_out, pipeline
