from itertools import combinations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

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




def isolated_forest_eval(df, top_k=50): 
    random_states = [0, 42, 123, 909]   
    anomaly_sets = {}   # To track the strict -1 flags
    topk_sets = {}      # To track the top_k most severe anomaly scores
    
    # 1. Run the models across different seeds
    for rs in random_states:
        # Assuming your isolated_forest function is defined elsewhere
        df_result, model = isolated_forest(
            df.copy(),                     
            n_estimators=230,
            max_samples=0.8,
            contamination=0.04,
            random_state=rs
        )
        
        # Capture indices where flag is explicitly -1
        anomaly_indices = df_result[df_result['Anomaly_Flag'] == -1].index 
        anomaly_sets[rs] = set(anomaly_indices)
        
        # Capture indices of the top K most anomalous scores
        # Note: Depending on your sklearn wrapper, lower scores or higher scores mean more anomalous.
        # If your function uses scikit-learn standard: lower (more negative) = more anomalous -> nsmallest is correct.
        topk_indices = df_result.nsmallest(top_k, "Anomaly_Score").index
        topk_sets[rs] = set(topk_indices)
     
    # 2. Evaluate Jaccard Similarity for Top-K Sets
    topk_scores = []
    print("--- Isolated Forest Top-K Jaccard Scores ---")
    for (rs1, set1), (rs2, set2) in combinations(topk_sets.items(), 2):
        if len(set1 | set2) == 0:
            jaccard = 0.0
        else:
            jaccard = len(set1 & set2) / len(set1 | set2)
        topk_scores.append(jaccard)
        print(f"Top-{top_k} Jaccard ({rs1}, {rs2}) = {jaccard:.3f}")
    
    # 3. Evaluate Jaccard Similarity for Flagged Sets (-1)
    flag_scores = []
    print("\n--- Isolated Forest Flagged Anomaly (-1) Jaccard Scores ---")
    for (rs1, set1), (rs2, set2) in combinations(anomaly_sets.items(), 2):
        if len(set1 | set2) == 0:
            jaccard = 0.0
        else:
            jaccard = len(set1 & set2) / len(set1 | set2)
        flag_scores.append(jaccard)
        print(f"Flag Jaccard ({rs1}, {rs2}) = {jaccard:.3f}")
        
    avg_topk_jaccard = np.mean(topk_scores)
    avg_flag_jaccard = np.mean(flag_scores) 
    
    return {
        "avg_topk_jaccard": avg_topk_jaccard,#How consistent the top 50 most anomalous scores are across different random seeds
        "avg_flag_jaccard": avg_flag_jaccard #How consistent the hard -1 flagged anomalies are across different random seeds
    }
