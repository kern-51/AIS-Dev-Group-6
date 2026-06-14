from itertools import combinations
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

def isolated_forest(df, model_config: dict):
    """Builds an Isolation Forest pipeline, fits it to the dataframe,

    and returns a copy of the dataframe with anomaly flags and scores,
    along with the trained pipeline object.
    """

    # Create a copy to avoid altering original data
    df_out = df.copy()
    
    # 1. Extract lists and params from config
    cols = model_config['columns']
    params = model_config['params']

    # 2. Setup preprocessing using config lists
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols['numeric']),
            ("ord", "passthrough", cols['ordinal']),
            ("nom", OneHotEncoder(handle_unknown="ignore"), cols['nominal']),
        ]
    )

    # 3. Define pipeline using config params
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("anomaly_detector", IsolationForest(
                n_estimators=params['n_estimators'],
                max_samples=params['max_samples'],
                contamination=params['contamination'],
                random_state=params['random_state'],
                n_jobs=-1
            )),
        ]
    )

    # 4. Fit and predict
    pipeline.fit(df_out)
    df_out["Anomaly_Flag"] = pipeline.predict(df_out)
    df_out["Anomaly_Score"] = pipeline.decision_function(df_out)

    return df_out, pipeline

from itertools import combinations
import numpy as np

def isolated_forest_eval(df, if_config, top_k=50): 
    # if_config is the dictionary for the isolation_forest model
    random_states = [0, 42, 123, 909]
    anomaly_sets = {}  # To track the strict -1 flags
    topk_sets = {}     # To track the top_k most severe anomaly scores
    
    # 1. Run the models across different seeds
    for rs in random_states:
        # Update the config for this specific run
        if_config['params']['random_state'] = rs
        
        # Call the isolated_forest function (ensure this is imported/defined)
        df_result, model = isolated_forest(df.copy(), if_config)
        
        # Capture indices where flag is explicitly -1
        anomaly_indices = df_result[df_result['Anomaly_Flag'] == -1].index 
        anomaly_sets[rs] = set(anomaly_indices)
        
        # Capture indices of the top K most anomalous scores
        # Lower (more negative) score = more anomalous
        topk_indices = df_result.nsmallest(top_k, "Anomaly_Score").index
        topk_sets[rs] = set(topk_indices)
    
    # Helper function to calculate Jaccard Similarity
    def get_jaccard(set1, set2):
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # 2. Evaluate Jaccard Similarity for Top-K Sets
    topk_scores = []
    print("\n--- Isolated Forest Top-K Jaccard Scores ---")
    for (rs1, set1), (rs2, set2) in combinations(topk_sets.items(), 2):
        score = get_jaccard(set1, set2)
        topk_scores.append(score)
        print(f"Top-{top_k} Jaccard ({rs1}, {rs2}) = {score:.3f}")
    
    # 3. Evaluate Jaccard Similarity for Flagged Sets (-1)
    flag_scores = []
    print("\n--- Isolated Forest Flagged Anomaly (-1) Jaccard Scores ---")
    for (rs1, set1), (rs2, set2) in combinations(anomaly_sets.items(), 2):
        score = get_jaccard(set1, set2)
        flag_scores.append(score)
        print(f"Flag Jaccard ({rs1}, {rs2}) = {score:.3f}")
        
    # 4. Return Averages
    return {
        "avg_topk_jaccard": np.mean(topk_scores) if topk_scores else 0,
        "avg_flag_jaccard": np.mean(flag_scores) if flag_scores else 0
    }