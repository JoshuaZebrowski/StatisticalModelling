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
print("\n\nSection 1.1\n")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"True Positive Rate (TPR): {tpr:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\n")


# Section 1.2: 5-Fold Cross-Validation
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Perform 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(LogisticRegression(), X, y, cv=skf, method='predict')

# Calculate evaluation metrics for cross-validation
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
accuracy_cv = accuracy_score(y, y_pred_cv)
precision_cv = precision_score(y, y_pred_cv)
recall_cv = recall_score(y, y_pred_cv)
f1_cv = f1_score(y, y_pred_cv)

# Calculate true positive rate (TPR) and false positive rate (FPR) for cross-validation
tn_cv, fp_cv, fn_cv, tp_cv = conf_matrix_cv.ravel()
tpr_cv = tp_cv / (tp_cv + fn_cv)  # True Positive Rate
fpr_cv = fp_cv / (fp_cv + tn_cv)  # False Positive Rate

# Print the results for cross-validation
print("Section 1.2\n")
print("Confusion Matrix:")
print(conf_matrix_cv)
print(f"Accuracy: {accuracy_cv:.4f}")
print(f"True Positive Rate (TPR): {tpr_cv:.4f}")
print(f"False Positive Rate (FPR): {fpr_cv:.4f}")
print(f"Precision: {precision_cv:.4f}")
print(f"Recall: {recall_cv:.4f}")
print(f"F1-Score: {f1_cv:.4f}")
print("\n")


# Section 1.3: Plot an ROC Curve for the Classifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute the ROC curve and AUC for the test set (from Section 1.1)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Brief Interpretation of Results
print("\nSection 1.3: Interpretation of the ROC Curve")
print(f"The ROC curve shows the trade-off between the true positive rate (TPR) and false positive rate (FPR) at various thresholds.")
print(f"The area under the curve (AUC) is {roc_auc:.2f}, which indicates the model's ability to distinguish between classes.")
print("An AUC closer to 1.0 indicates a better-performing model, while an AUC of 0.5 suggests random guessing.")