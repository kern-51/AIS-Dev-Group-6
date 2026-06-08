import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
def train_gradient_boosting(df):
  """
  Builds a gradient boosting model using data from the pipeline, 
  and returns the evaluation metrics the model on training and testing datasets.
  """
  X = df.drop('Activity Level', axis=1) 
  y = df['Activity Level']
  #
  # split data for training and testing
  #

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

  target_names = ['1.0', '2.0', '3.0']

  model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  print("Classification Report for Test Set \n")
  print(classification_report(y_test, y_pred, target_names=target_names), "\n")

  y_pred_train = model.predict(X_train)
  print("Classification Report for Train Set \n")
  print(classification_report(y_train, y_pred_train, target_names=target_names))
  return model
