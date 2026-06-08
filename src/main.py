from src.clean import data_clean
from src.feature_engineering import engineer_features
from src.ingest import data_ingest
from src.random_forest import train_random_forest
from src.gradient_boosting import train_gradient_boosting
from src.isolated_forest import isolated_forest
from src.isolated_forest import isolated_forest_eval
def pipeline():
    df = data_ingest()
    df = data_clean(df)
    df = engineer_features(df)
    train_random_forest(df)
    train_gradient_boosting(df)
    anomaly_df = isolated_forest(df, contamination=0.05, random_state=42)
    isolated_forest_eval(df)

    

if __name__ == "__main__":
    pipeline()
