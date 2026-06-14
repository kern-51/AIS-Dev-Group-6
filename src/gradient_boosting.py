import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
def train_gradient_boosting(X_train, X_test, y_train, y_test, target_names, gb_config: dict):
    """
    Builds a gradient boosting model using data from the pipeline, 
    and returns the evaluation metrics the model on training and testing datasets.
    """
    # 1. Setup Resampling (Explicitly pull the key you need)
    rus = RandomOverSampler(random_state=gb_config['resampling_random_state'])
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # 2. Setup Model (Explicitly pull the keys you need)
    model = GradientBoostingClassifier(
        n_estimators=gb_config['n_estimators'],
        learning_rate=gb_config['learning_rate'],
        random_state=gb_config['model_random_state']
    )

    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    
    print("Gradient Boosting Classification Report for Test Set\n")
    print(classification_report(y_test, y_pred, target_names=target_names), "\n")

    y_pred_train = model.predict(X_train)
    print("Gradient Boosting Classification Report for Train Set\n")
    print(classification_report(y_train, y_pred_train, target_names=target_names))
    return model