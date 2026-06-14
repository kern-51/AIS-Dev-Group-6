from src.clean import data_clean
from src.feature_engineering import engineer_features
from src.ingest import data_ingest
from src.random_forest import train_random_forest
from src.gradient_boosting import train_gradient_boosting
from src.isolated_forest import isolated_forest
from src.isolated_forest import isolated_forest_eval
from sklearn.model_selection import train_test_split
import yaml


def pipeline():
    cfg = yaml.safe_load(open("config.yaml", "r"))
    rf_config = cfg['models']['random_forest']
    gb_config = cfg['models']['gradient_boosting']
    if_config = cfg['models']['isolation_forest']
    df = data_ingest()
    df = data_clean(df)
    df = engineer_features(df)

    X = df.drop('Activity Level', axis=1) 
    y = df['Activity Level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    target_names = ['1.0', '2.0', '3.0']

    train_random_forest(X_train, X_test, y_train, y_test, target_names, rf_config)
    train_gradient_boosting(X_train, X_test, y_train, y_test, target_names, gb_config)
    anomaly_df = isolated_forest(df, if_config)
    isolated_forest_eval(df, if_config)

if __name__ == "__main__":
    pipeline()
