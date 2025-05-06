import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


print("\n\n\nSection 2.1\n")
# Load the dataset
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\40402274_features.csv'
data = pd.read_csv(file_path)

# Filter the dataset for the required classes: "a", "j", smiley, sad, and xclaim
classes_to_include = {'a', 'j', 'smiley', 'sad', 'xclaim'}
data = data[data['LABEL'].isin(classes_to_include)]

# Encode the target labels as integers
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['LABEL'])

# Select 4 features for classification
# Features chosen: 'nr_pix', 'aspect_ratio', 'rows_with_3p', 'no_neigh_horiz'
features = ['nr_pix', 'aspect_ratio', 'rows_with_3p', 'no_neigh_horiz']
X = data[features]
y = data['class']

# Perform KNN classification for odd values of k between 1 and 13
k_values = range(1, 14, 2)
accuracies = []

for k in k_values:
    # Initialize and fit the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # Predict on the training data (since no separate test set is used)
    y_pred = knn.predict(X)
    
    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    accuracies.append((k, accuracy))

# Report the accuracies for all values of k
print("\nSummary of Accuracies:")
for k, accuracy in accuracies:
    print(f"k = {k}: Accuracy = {accuracy:.4f}")




from sklearn.model_selection import cross_val_score

# Section 2.2: Perform KNN classification with 5-fold cross-validation
print("\n\nSection 2.2\n")
cv_accuracies = []

for k in k_values:
    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Perform 5-fold cross-validation and calculate the mean accuracy
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    cv_accuracies.append((k, mean_accuracy))

# Report the cross-validated accuracies for all values of k
print("\nSummary of Cross-Validated Accuracies:")
for k, mean_accuracy in cv_accuracies:
    print(f"k = {k}: Cross-Validated Accuracy = {mean_accuracy:.4f}")




from sklearn.metrics import confusion_matrix, classification_report

# Section 2.3: Confusion Matrix for the Best k
# Find the best k (highest cross-validated accuracy)
best_k, best_accuracy = max(cv_accuracies, key=lambda item: item[1])
print(f"\n\n\nSection 2.3\n\nBest k = {best_k} with Cross-Validated Accuracy = {best_accuracy:.4f}")

# Train the KNN model with the best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X, y)

# Predict on the full dataset
y_pred_best = best_knn.predict(X)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y, y_pred_best)

# Get the class names from the label encoder
class_names = label_encoder.classes_

# Convert the confusion matrix to a DataFrame for better readability
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Print the confusion matrix with class names
print("\nConfusion Matrix (with Class Names):")
print(conf_matrix_df)

classes = ['a', 'j', 'sad', 'smiley', 'xclaim']

# Find top misclassified pairs
errors = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and conf_matrix[i, j] > 0:
            errors.append((classes[i], classes[j], conf_matrix[i, j]))

# Sort by error count
errors.sort(key=lambda x: -x[2])

print("\nMost confused pairs:")
for true, pred, count in errors:
    print(f"{true} → {pred}: {count} errors")



# Section 2.4: Plotting Training and Cross-Validated Error Rates
print("\n\n\nSection 2.4\n")
import matplotlib.pyplot as plt

# Convert accuracies to error rates
training_error_rates = [1 - acc for _, acc in accuracies]
cv_error_rates = [1 - acc for _, acc in cv_accuracies]

# Convert k values to 1/k for the x-axis
inverse_k = [1 / k for k, _ in accuracies]

# Plot the error rates
plt.figure(figsize=(8, 6))
plt.plot(inverse_k, training_error_rates, marker='o', label='Training Error Rate', color='blue')
plt.plot(inverse_k, cv_error_rates, marker='o', label='Cross-Validated Error Rate', color='orange')
plt.axhline(y=min(cv_error_rates), color='black', linestyle='--', label='Minimum CV Error Rate')

# Add labels, title, and legend
plt.xlabel('1/k')
plt.ylabel('Error Rate')
plt.title('Training and Cross-Validated Error Rates vs. 1/k')
plt.legend()
plt.grid()
plt.show()

# Brief Interpretation of Results
print("\nInterpretation:")
print("1. The training error rate decreases as k increases, as the model becomes less sensitive to noise.")
print("2. The cross-validated error rate initially decreases and then increases, indicating overfitting for small k and underfitting for large k.")
print("3. The optimal value of k minimizes the cross-validated error rate, balancing the trade-off between bias and variance.")