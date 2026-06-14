from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_random_forest(df, target_column="Activity Level"):
    """
    Trains and evaluates a Random Forest classification model using the
    processed gas monitoring dataset.
    The model uses environmental sensor readings and engineered features
    to predict Activity Level. The dataset is split into training and
    testing sets, after which the Random Forest model is trained and used
    to generate predictions.
    Model performance is evaluated using a classification report that
    includes Precision, Recall, and F1-Score for each activity level
    class. These metrics help assess how effectively the model can
    classify environmental activity levels.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    target_names = ['1.0', '2.0', '3.0']

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

    print("Random Forest Classification Report for Test Set\n")
    print(classification_report(y_test, y_pred, target_names=target_names), "\n")

    y_pred_train = model.predict(X_train)
    print("Random Forest Classification Report for Train Set\n")
    print(classification_report(y_train, y_pred_train, target_names=target_names))

    return model
