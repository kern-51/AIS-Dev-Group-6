import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
def train_gradient_boosting(X_train, X_test, y_train, y_test, target_names):
  """
  Builds a gradient boosting model using data from the pipeline, 
  and returns the evaluation metrics the model on training and testing datasets.
  """
  rus = RandomOverSampler(random_state=42)
  X_resampled, y_resampled = rus.fit_resample(X_train, y_train)



  model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42)

  model.fit(X_resampled, y_resampled)

  y_pred = model.predict(X_test)

  print("Gradient Boosting Classification Report for Test Set\n")
  print(classification_report(y_test, y_pred, target_names=target_names), "\n")

  y_pred_train = model.predict(X_train)
  print("Gradient Boosting Classification Report for Train Set\n")
  print(classification_report(y_train, y_pred_train, target_names=target_names))
  return model