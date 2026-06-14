from itertools import combinations, product
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest


def isolated_forest(df, model_config: dict):
    """Builds an Isolation Forest pipeline from a config dict,
    fits it to the dataframe, and returns a copy with anomaly flags
    and scores, along with the trained pipeline object.
    """
    df_out = df.copy()

    # Extract from config
    cols   = model_config['columns']
    params = model_config['params']

    # Preprocessing — ordinal uses StandardScaler to prevent feature dominance
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols['numeric']),
            ("ord", StandardScaler(), cols['ordinal']),   # ← StandardScaler, not passthrough
            ("nom", OneHotEncoder(handle_unknown="ignore"), cols['nominal']),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("anomaly_detector", IsolationForest(
                n_estimators =params['n_estimators'],
                max_samples  =params['max_samples'],
                max_features =params['max_features'],   # ← supported
                contamination=params['contamination'],
                random_state =params['random_state'],   # ← clean arg, not mutated
                n_jobs=-1,
            )),
        ]
    )

    pipeline.fit(df_out)
    df_out["Anomaly_Flag"]  = pipeline.predict(df_out)
    df_out["Anomaly_Score"] = pipeline.decision_function(df_out)

    return df_out, pipeline





def isolated_forest_eval(df, model_config: dict):
    random_states     = model_config['params']['test_random_states']
    n_estimators_list = model_config['params']['n_estimators_list']
    max_samples_list  = model_config['params']['test_max_samples']
    max_features_list = model_config['params']['test_max_features']

    contamination = model_config['params']['test_contamination']
    #contamination is a list, use the first value
    contamination = contamination[0]

    top_k = int(contamination * len(df))
    print(f"top_k = {top_k} ({contamination*100:.1f}% of {len(df)} rows)\n")

    results_list   = []
    best_snapshot  = None   # ← holds top20 only for the best config

    # 36 configs (3 × 4 × 3)
    for n_est, max_samp, max_feat in product(
        n_estimators_list, max_samples_list, max_features_list
    ):
        anomaly_sets = {}
        topk_sets    = {}
        anomaly_pcts = []
        seed42_snapshot = None   # ← snapshot within this config, not printed yet

        for rs in random_states:
            run_config = {
                'columns': model_config['columns'],
                'params': {
                    **model_config['params'],
                    'n_estimators' : n_est,
                    'max_samples'  : max_samp,
                    'max_features' : max_feat,
                    'random_state' : rs,
                }
            }

            df_result, _ = isolated_forest(df.copy(), run_config)

            anomaly_indices  = df_result[df_result['Anomaly_Flag'] == -1].index
            anomaly_sets[rs] = set(anomaly_indices)
            anomaly_pcts.append(len(anomaly_indices) / len(df_result) * 100)

            topk_sets[rs] = set(df_result.nsmallest(top_k, "Anomaly_Score").index)

            # Store snapshot quietly — only used if this turns out to be best config
            if rs == 42:
                seed42_snapshot = df_result.sort_values("Anomaly_Score").copy()

        # Jaccard scores
        topk_scores = [
            len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
            for (_, s1), (_, s2) in combinations(topk_sets.items(), 2)
        ]
        flag_scores = [
            len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0
            for (_, s1), (_, s2) in combinations(anomaly_sets.items(), 2)
        ]

        avg_topk = np.mean(topk_scores)
        avg_flag = np.mean(flag_scores)
        avg_pct  = np.mean(anomaly_pcts)

        results_list.append({
            "n_estimators"    : n_est,
            "max_samples"     : str(max_samp),
            "max_features"    : max_feat,
            "avg_topk_jaccard": round(avg_topk, 3),
            "avg_flag_jaccard": round(avg_flag, 3),
            "avg_anomaly_pct" : round(avg_pct, 2),
            "_snapshot"       : seed42_snapshot,   # ← carry snapshot alongside, not printed
        })
    print(f"results_list length: {len(results_list)}")
    # Build and sort DataFrame
    df_results = (
        pd.DataFrame(results_list)
        .sort_values("avg_topk_jaccard", ascending=False)
        .reset_index(drop=True)
    )

    # Pull best snapshot before dropping the column
    best_snapshot = df_results.iloc[0]["_snapshot"]
    df_results    = df_results.drop(columns=["_snapshot"])

    # Group-level insights
    print("--- Avg Top-K Jaccard by max_features ---")
    print(df_results.groupby("max_features")["avg_topk_jaccard"].mean().round(3))

    print("\n--- Avg Top-K Jaccard by max_samples ---")
    print(df_results.groupby("max_samples")["avg_topk_jaccard"].mean().round(3))

    print("\n--- Avg Top-K Jaccard by n_estimators ---")
    print(df_results.groupby("n_estimators")["avg_topk_jaccard"].mean().round(3))

    # Best config summary
    best = df_results.iloc[0]
    print(f"\n✅ Best config by Top-K Jaccard:")
    print(f"   n_estimators : {best['n_estimators']}")
    print(f"   max_samples  : {best['max_samples']}")
    print(f"   max_features : {best['max_features']}")
    print(f"   Top-K Jaccard: {best['avg_topk_jaccard']}")
    print(f"   Flag Jaccard : {best['avg_flag_jaccard']}")
    print(f"   Anomaly %    : {best['avg_anomaly_pct']}")

    # Top 20 anomalous rows — only for best config
    if best_snapshot is not None:
        sense_cols   = [
            "Anomaly_Score", "Anomaly_Flag",
            "Temperature", "Humidity",
            "CO2_InfraredSensor", "CO_GasSensor",
            "Activity Level", "HVAC Operation Mode"
        ]
        display_cols = [c for c in sense_cols if c in best_snapshot.columns]

        print(f"\n--- Top 5 Most Anomalous Rows (best config, seed=42) ---")
        top20 = best_snapshot.head(5)[display_cols]
        print(top20.to_string(index=True))

        # Anomaly explanation
        print(f"\n--- Anomaly Explanation (best config, seed=42) ---")

        # Only keep display_cols that exist in the original df (excludes Anomaly_Score, Anomaly_Flag)
        mean_cols    = [c for c in display_cols if c in df.columns]
        overall_means = df[mean_cols].select_dtypes(include="number").mean()
        overall_stds  = df[mean_cols].select_dtypes(include="number").std()

        for idx, row in top20.iterrows():
            reasons = []
            for col in display_cols:
                if col in ["Anomaly_Score", "Anomaly_Flag"]:
                    continue
                if col not in overall_means:
                    continue
                mean_val = overall_means[col]
                std_val  = overall_stds[col]
                row_val  = row[col]
                z        = (row_val - mean_val) / (std_val + 1e-9)

                if abs(z) >= 2.0:   # flag if 2+ std devs from mean
                    direction = "HIGH" if z > 0 else "LOW"
                    reasons.append(f"{col}={row_val:.2f} ({direction}, z={z:.2f}, mean={mean_val:.2f})")

            reason_str = " | ".join(reasons) if reasons else "No single dominant feature — multivariate anomaly"
            print(f"  Row {idx:>5} | Score={row['Anomaly_Score']:.4f} → {reason_str}")

    # Styled DataFrame
    styled = (
        df_results.style
        .background_gradient(subset=["avg_topk_jaccard"], cmap="Greens")
        .background_gradient(subset=["avg_flag_jaccard"], cmap="Blues")
        .background_gradient(subset=["avg_anomaly_pct"],  cmap="Oranges")
        .format({
            "avg_topk_jaccard": "{:.3f}",
            "avg_flag_jaccard": "{:.3f}",
            "avg_anomaly_pct" : "{:.2f}%",
        })
        .set_caption(f"Isolation Forest Sweep — top_k={top_k} | contamination={contamination}")
    )
    return styled

