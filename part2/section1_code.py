import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve


#-----------------------------------------------------------------------------------------------------------------#

# 1.1: Splitting Data & Binary Classification with Logistic Regression

# loads the dataset
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\40402274_features.csv'
data = pd.read_csv(file_path)

# defines the features and target
features = ['nr_pix', 'aspect_ratio']
target = 'is_letter'

# ensures the dataset requires the required classes
letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
data['is_letter'] = data['LABEL'].apply(lambda x: 1 if x in letters else 0)

# randomly splits the dataset into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fitting the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# predicting the outcomes for the test set
y_prob = logistic_model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# calculating evaluation metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# calculating true positive rate (tpr) and false positive rate (fpr)
tn, fp, fn, tp = conf_matrix.ravel()
tpr = tp / (tp + fn)  
fpr = fp / (fp + tn)  

# printing results to console
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

#-----------------------------------------------------------------------------------------------------------------#

# Section 1.2: Repeating using 5-Fold Cross-Validation

from sklearn.model_selection import cross_val_predict, StratifiedKFold

# peforming the 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(LogisticRegression(), X, y, cv=skf, method='predict')

# calculating evaluation metrics for cross-validation
conf_matrix_cv = confusion_matrix(y, y_pred_cv)
accuracy_cv = accuracy_score(y, y_pred_cv)
precision_cv = precision_score(y, y_pred_cv)
recall_cv = recall_score(y, y_pred_cv)
f1_cv = f1_score(y, y_pred_cv)

# calculating true positive rate (tpr) and false positive rate (fpr) for cross-validation
tn_cv, fp_cv, fn_cv, tp_cv = conf_matrix_cv.ravel()
tpr_cv = tp_cv / (tp_cv + fn_cv)  
fpr_cv = fp_cv / (fp_cv + tn_cv)  

# printing results to console
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

#-----------------------------------------------------------------------------------------------------------------#

# Section 1.3: Plot an ROC Curve for the Classifier

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# calculating predicted probabilities using cross-validation
y_prob_cv = cross_val_predict(LogisticRegression(), X, y, cv=skf, method='predict_proba')[:, 1]

# calculating the ROC curve based on cross-validation results
fpr_cv, tpr_cv, thresholds_cv = roc_curve(y, y_prob_cv)
roc_auc_cv = auc(fpr_cv, tpr_cv)

# plotting the ROC curve
plt.figure()
plt.plot(fpr_cv, tpr_cv, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_cv:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Cross-Validation)')
plt.legend(loc='lower right')
plt.grid()
plt.show()
