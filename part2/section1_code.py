import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve

# Load the dataset
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\40402274_features.csv'
data = pd.read_csv(file_path)

# Define features and target
features = ['nr_pix', 'aspect_ratio']
target = 'is_letter'

# Ensure the target column exists
letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
data['is_letter'] = data['LABEL'].apply(lambda x: 1 if x in letters else 0)

# Split the data into training and test sets (80% train, 20% test)
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict probabilities and classes on the test set
y_prob = logistic_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# Calculate evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate true positive rate (TPR) and false positive rate (FPR)
tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn)  # True Positive Rate
fpr = fp / (fp + tn)  # False Positive Rate

# Print the results
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")