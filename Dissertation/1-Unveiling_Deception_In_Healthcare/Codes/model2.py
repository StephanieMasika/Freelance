import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load your dataset (replace with your own data)
df = pd.read_csv(r'C:\Users\masik\OneDrive\Desktop\Dissertation\HEALTHCARE PROVIDER FRAUD DETECTION ANALYSIS\HEALTHCARE PROVIDER FRAUD DETECTION ANALYSIS\Train_Beneficiarydata.csv')

# Example: Features and label
X = df.drop('County', axis=1)  # Assuming 'fraud' is the label column
y = df['County']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=[np.number]).columns

# Preprocessing for numeric data: StandardScaler
# Preprocessing for categorical data: OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Create a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Initialize the Isolation Forest model
iso_forest = IsolationForest(contamination=0.05, random_state=42)

# Train the model on preprocessed data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

iso_forest.fit(X_train_preprocessed)

# Predict on the test set
y_pred_unsupervised = iso_forest.predict(X_test_preprocessed)

# In Isolation Forest, -1 indicates anomaly (potential fraud), and 1 indicates normal
y_pred_unsupervised = np.where(y_pred_unsupervised == -1, 1, 0)

# Evaluate the model
accuracy_unsupervised = accuracy_score(y_test, y_pred_unsupervised)
precision_unsupervised = precision_score(y_test, y_pred_unsupervised)
recall_unsupervised = recall_score(y_test, y_pred_unsupervised)
f1_unsupervised = f1_score(y_test, y_pred_unsupervised)
conf_matrix_unsupervised = confusion_matrix(y_test, y_pred_unsupervised)

print(f"Unsupervised - Accuracy: {accuracy_unsupervised}")
print(f"Unsupervised - Precision: {precision_unsupervised}")
print(f"Unsupervised - Recall: {recall_unsupervised}")
print(f"Unsupervised - F1 Score: {f1_unsupervised}")
print(f"Unsupervised - Confusion Matrix:\n{conf_matrix_unsupervised}")
