from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(df, target_column="Activity Level"):
    """
    Taken the ingested data from the pipeline, built a random forest model using data to predict
    activity level.  Output is the evauluation such as the accuracy, recall, precision and F1 score.
    This is done to see how well the model performs with the data. Summary is that it splits
    the data, trains the model and predicts and evaluates the results.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred)

    return model, accuracy, report
