from src.clean import data_clean
from src.feature_engineering import engineer_features
from src.ingest import data_ingest
from src.random_forest import train_random_forest
from src.logistic_regression import train_logistic_regression
from src.isolated_forest import isolated_forest
def pipeline():
    df = data_ingest()
    df = data_clean(df)
    df = engineer_features(df)
    train_logistic_regression(df)
    train_logistic_regression(df)
    isolated_forest(df, contamination=0.05, random_state=42)

if __name__ == "__main__":
    pipeline()
