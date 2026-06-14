from itertools import combinations, product
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest


def isolated_forest(
    df, n_estimators=100, max_samples="auto", contamination=0.05, random_state=42, max_features=1.0
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
    'Ambient Light Level',
    'Activity Level'
    ]
    nominal_cols = [
    'Time of Day',    'HVAC Operation Mode', # Time is in nominal as it is cyclical, ie evening is not > morning
    ]

    # Setup the preprocessing layers
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols), #scaling to prevent Feature Dominance
        ("ord", StandardScaler(), ordinal_cols), #prevents the ordinal ranks from carrying unintended mathematical weight compared to the rest of the dataset (ie Feature Dominance)
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols), # no relations
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
                    max_features=max_features,
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





def isolated_forest_eval(df, contamination=0.05): #testing range of params
    random_states  = [0, 42, 123, 909]
    n_estimators_list  = [100, 200, 500]
    max_samples_list   = ["auto", 0.5, 0.8, 1.0]
    max_features_list  = [0.5, 0.75, 1.0]

    # top_k tied to contamination rate
    top_k = int(contamination * len(df))
    print(f"top_k = {top_k} ({contamination*100:.1f}% of {len(df)} rows)\n")

    all_results = {}

    # Sweep all hyperparameter combinations
    for n_est, max_samp, max_feat in product(
        n_estimators_list, max_samples_list, max_features_list
    ):
        config_key = f"n_est={n_est}_samp={max_samp}_feat={max_feat}"
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_key}")
        print(f"{'='*60}")

        anomaly_sets    = {}
        topk_sets       = {}
        anomaly_pcts    = []
        top20_snapshots = {}

        # 1. Run across different seeds
        for rs in random_states:
            df_result, _ = isolated_forest(
                df.copy(),
                n_estimators=n_est,
                max_samples=max_samp,
                max_features=max_feat,
                contamination=contamination,
                random_state=rs
            )

            # Flagged anomalies (-1)
            anomaly_indices      = df_result[df_result['Anomaly_Flag'] == -1].index
            anomaly_sets[rs]     = set(anomaly_indices)

            # Anomaly percentage
            n_anomalies = len(anomaly_indices)
            pct         = n_anomalies / len(df_result) * 100
            anomaly_pcts.append(pct)
            print(f"  seed={rs} → {n_anomalies} anomalies ({pct:.2f}%)")

            # Top-K by score (main stability metric)
            topk_indices     = df_result.nsmallest(top_k, "Anomaly_Score").index
            topk_sets[rs]    = set(topk_indices)

            # Sense-check snapshot from seed=42
            if rs == 42:
                top20_snapshots[rs] = df_result.sort_values("Anomaly_Score").head(20)

        # 2. Top-K Jaccard — main stability metric
        topk_scores = []
        print(f"\n  --- Top-{top_k} Jaccard (main metric) ---")
        for (rs1, s1), (rs2, s2) in combinations(topk_sets.items(), 2):
            jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
            topk_scores.append(jaccard)
            print(f"  seed {rs1} vs {rs2} = {jaccard:.3f}")

        # 3. Flag Jaccard — secondary metric
        flag_scores = []
        print(f"\n  --- Flag (-1) Jaccard ---")
        for (rs1, s1), (rs2, s2) in combinations(anomaly_sets.items(), 2):
            jaccard = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
            flag_scores.append(jaccard)
            print(f"  seed {rs1} vs {rs2} = {jaccard:.3f}")

        avg_topk = np.mean(topk_scores)
        avg_flag = np.mean(flag_scores)
        avg_pct  = np.mean(anomaly_pcts)

        print(f"\n  Avg anomaly %          : {avg_pct:.2f}%")
        print(f"  Avg Top-{top_k} Jaccard: {avg_topk:.3f}  ← main stability metric")
        print(f"  Avg Flag Jaccard       : {avg_flag:.3f}")

        # 4. Sense-check: top 20 most anomalous rows (seed=42)
        if 42 in top20_snapshots:
            sense_cols   = [
                "Anomaly_Score", "Temperature", "Humidity",
                "CO2_InfraredSensor", "CO_GasSensor",
                "Activity Level", "HVAC Operation Mode"
            ]
            display_cols = [c for c in sense_cols if c in top20_snapshots[42].columns]
            print(f"\n  --- Top 20 Most Anomalous Rows (seed=42) ---")
            print(top20_snapshots[42][display_cols].to_string(index=True))

        all_results[config_key] = {
            "n_estimators"     : n_est,
            "max_samples"      : max_samp,
            "max_features"     : max_feat,
            "avg_topk_jaccard" : avg_topk,   # ← main metric
            "avg_flag_jaccard" : avg_flag,
            "avg_anomaly_pct"  : avg_pct,
        }

    # 5. Summary table — sorted by Top-K Jaccard descending
    print(f"\n{'='*60}")
    print("SUMMARY — sorted by Avg Top-K Jaccard (best configs first)")
    print(f"{'='*60}")

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["avg_topk_jaccard"],
        reverse=True
    )

    print(f"{'Config':<40} {'Anomaly %':<12} {'Top-K Jaccard':<16} {'Flag Jaccard'}")
    for config, res in sorted_results:
        print(
            f"{config:<40} "
            f"{res['avg_anomaly_pct']:<12.2f}"
            f"{res['avg_topk_jaccard']:<16.3f}"
            f"{res['avg_flag_jaccard']:.3f}"
        )

    # 6. Best config
    best_config, best_res = sorted_results[0]
    print(f"\n✅ Best config by Top-K Jaccard: {best_config}")
    print(f"   Top-K Jaccard : {best_res['avg_topk_jaccard']:.3f}")
    print(f"   Flag Jaccard  : {best_res['avg_flag_jaccard']:.3f}")
    print(f"   Anomaly %     : {best_res['avg_anomaly_pct']:.2f}%")

    return all_results


# Usage
#results = isolated_forest_eval(df, contamination=0.05)
