import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


#-----------------------------------------------------------------------------------------------------------------#

# Section 2.1: KNN Classification with Odd Values of k
print("\n\n\nSection 2.1\n")

# loads the dataset
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\40402274_features.csv'
data = pd.read_csv(file_path)

# filters the dataset to include only the specified classes
classes_to_include = {'a', 'j', 'smiley', 'sad', 'xclaim'}
data = data[data['LABEL'].isin(classes_to_include)]

# encodes the class labels to integers
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['LABEL'])

# chooses the features for KNN classification
features = ['nr_pix', 'aspect_ratio', 'rows_with_3p', 'no_neigh_horiz']
X = data[features]
y = data['class']

# carries out KNN classification for odd values of k from 1 to 13
k_values = range(1, 14, 2)
accuracies = []

for k in k_values:
    # initialises the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    
    # predicts the class labels for the training set
    y_pred = knn.predict(X)
    
    # calculates the accuracy
    accuracy = accuracy_score(y, y_pred)
    accuracies.append((k, accuracy))

# reports the accuracies for all values of k
print("\nSummary of Accuracies:")
for k, accuracy in accuracies:
    print(f"k = {k}: Accuracy = {accuracy:.4f}")




#-----------------------------------------------------------------------------------------------------------------#

from sklearn.model_selection import cross_val_score

# Section 2.2: Perform KNN classification with 5-fold cross-validation
print("\n\nSection 2.2\n")
cv_accuracies = []

for k in k_values:
    # initialises the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # performs 5-fold cross-validation and calculates the mean accuracy
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    mean_accuracy = scores.mean()
    cv_accuracies.append((k, mean_accuracy))

# reports the cross-validated accuracies for all values of k
print("\nSummary of Cross-Validated Accuracies:")
for k, mean_accuracy in cv_accuracies:
    print(f"k = {k}: Cross-Validated Accuracy = {mean_accuracy:.4f}")




#-----------------------------------------------------------------------------------------------------------------#

from sklearn.metrics import confusion_matrix

# Section 2.3: Confusion Matrix for the Best k

# finds the best k based on cross-validated accuracy
best_k, best_accuracy = max(cv_accuracies, key=lambda item: item[1])
print(f"\n\n\nSection 2.3\n\nBest k = {best_k} with Cross-Validated Accuracy = {best_accuracy:.4f}")

# trains the KNN classifier with the best k
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X, y)

# predicts the class labels for the training set
y_pred_best = best_knn.predict(X)

# computes the confusion matrix
conf_matrix = confusion_matrix(y, y_pred_best)

# gets the class names from the label encoder
class_names = label_encoder.classes_

# converts the confusion matrix to a DataFrame for better readability
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# prints the confusion matrix and classification report
print("\nConfusion Matrix (with Class Names):")
print(conf_matrix_df)

classes = ['a', 'j', 'sad', 'smiley', 'xclaim']

# finds the most confused pairs of classes
errors = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and conf_matrix[i, j] > 0:
            errors.append((classes[i], classes[j], conf_matrix[i, j]))

# sorts the errors by the number of errors in descending order
errors.sort(key=lambda x: -x[2])

print("\nMost confused pairs:")
for true, pred, count in errors:
    print(f"{true} → {pred}: {count} errors")


#-----------------------------------------------------------------------------------------------------------------#

import matplotlib.pyplot as plt

# Section 2.4: Plotting Training and Cross-Validated Error Rates
print("\n\n\nSection 2.4\n")

# calculate error rates
training_error_rates = [1 - acc for _, acc in accuracies]
cv_error_rates = [1 - acc for _, acc in cv_accuracies]

# convert k values to 1/k for plotting
inverse_k = [1 / k for k, _ in accuracies]

# plots the training and cross-validated error rates
plt.figure(figsize=(8, 6))
plt.plot(inverse_k, training_error_rates, marker='o', label='Training Error Rate', color='blue')
plt.plot(inverse_k, cv_error_rates, marker='o', label='Cross-Validated Error Rate', color='orange')
plt.axhline(y=min(cv_error_rates), color='black', linestyle='--', label='Minimum CV Error Rate')

# add labels and title
plt.xlabel('1/k')
plt.ylabel('Error Rate')
plt.title('Training and Cross-Validated Error Rates vs. 1/k')
plt.legend()
plt.grid()
plt.show()
