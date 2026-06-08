Group-6

Members: 
Ivo Lim, Qian HongHeng, Soon You Kern

Who wrote which .py files (each member must write at least 1 py file that can be used for code quality assessment)
ingest.py - kern
clean.py - kern
feature_engineering.py - ivo
isolated_forest.py - kern
gradient_boosting.py - hongheng

• Instructions on how to run the pipeline
Git clone the files.
• Instructions on how start your docker development environment (if any)
Open wsl and docker desktop.
Go to the directory where the files are at.
Run docker compose up to star the environment
Run docker compose down --rmi all to close everything.







• Summary of key findings of EDA
The Exploratory Data Analysis (EDA) revealed several important patterns within the gas monitoring dataset. Environmental variables such as temperature, humidity, carbon dioxide (CO₂), and metal oxide sensor readings showed noticeable variation across different activity levels. Higher activity levels generally corresponded with increased gas concentration and environmental fluctuations, suggesting a relationship between human activity and sensor measurements. Correlation analysis also indicated that some sensor readings were related to one another, which provided opportunities for feature engineering. Overall, the EDA demonstrated that environmental sensor data contains meaningful information that can be used to classify activity levels and monitor environmental conditions effectively.

• Explain and justify features that are engineered
To improve model performance, several engineered features were created from the original sensor readings. These included average metal oxide readings and average CO₂ measurements obtained from multiple sensors. Aggregating values from multiple sensors helps reduce noise and provides a more stable representation of environmental conditions. Additional derived features were created to capture broader environmental patterns rather than relying on individual sensor values alone. These engineered features were expected to improve predictive performance by providing the machine learning models with more meaningful and representative inputs of the monitored environment.

• Explanation of choice of models (train at least 3 models) and justify any tuning methods used
Random Forest was selected as the primary classification model because it performs well on structured tabular data and can capture complex relationships between environmental sensor readings and activity levels. The model combines multiple decision trees, reducing the risk of overfitting while improving prediction accuracy. Random Forest is also capable of handling both original and engineered features effectively and provides feature importance information, making the model more interpretable. Hyperparameter tuning was performed by adjusting parameters such as the number of trees and maximum tree depth to achieve better classification performance.
Isolation Forest was chosen as an anomaly detection model to identify unusual environmental conditions that may indicate abnormal sensor behaviour or unexpected environmental events. Unlike Random Forest, Isolation Forest does not require labelled data and works by isolating observations that differ significantly from the majority of the dataset. This makes it suitable for environmental monitoring systems where detecting abnormal readings is as important as classifying normal activity levels. The contamination parameter was tuned to control the expected proportion of anomalies and improve anomaly detection performance.

• Explain any specific choice of metrics that are important to the problem statement
Accuracy was selected as the primary evaluation metric because the objective of the project is to correctly classify activity levels based on environmental sensor data. In addition, Precision, Recall, and F1-Score were used to provide a more comprehensive assessment of model performance. Precision measures how many predicted instances are actually correct, while Recall evaluates the model's ability to identify all relevant instances. F1-Score balances both Precision and Recall, making it useful when evaluating overall classification quality. Using multiple metrics ensures that the models are evaluated from different perspectives rather than relying solely on overall accuracy.
