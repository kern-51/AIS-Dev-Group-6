
from preprocessing import clean_data
from imputation import run_knn_imputer
from model import train_model

def pipeline():
    df = data_ingest("raw_data.csv")
    df = data_clean(df)
    df = run_knn_imputer(df, k=7)
    train_model(df)

if __name__ == "__main__":
    pipeline()
