

def isolated_forest(
    df, n_estimators=100, max_samples="auto", contamination=0.05, random_state=42
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
    'MetalOxideSensor_Unit2' ,
    'MetalOxideSensor_Unit3',
    'MetalOxideSensor_Unit4' ,
    "metal_oxide_avg",
    "co2_avg",
    "co2_sensor_difference",
    "gas_intensity_score"
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
                    n_estimators=n_estimators,  # <-- Pass it here dynamically
                    max_samples=max_samples,    # <-- Pass it here dynamically
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




def isolated_forest_eval(df): #scoring how outliers overlap between different
    random_states = [0, 42, 123, 909]   # your list of seeds
    anomaly_sets = {}   # store indices (or boolean masks) for each run

    for rs in random_states:
        df_result, model = isolated_forest(
            df.copy(),                     # work on a copy to avoid overwriting
            n_estimators=230,
            max_samples=0.8,
            contamination=0.04,
            random_state=rs
        )
        # Assuming df_result has an 'anomaly' column: -1 = outlier, 1 = normal
        anomaly_indices = df_result[df_result['Anomaly_Flag'] == -1].index #use flag to find specific outliers not score given to outliers
        anomaly_sets[rs] = set(anomaly_indices)
     
    avg_jaccard = np.mean(list(jaccard_scores.values()))
    score = avg_jaccard:.3f
    return score
