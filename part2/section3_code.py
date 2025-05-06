import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

# Section 3.1: Random Forest Grid Search
print("\nSection 3.1: Random Forest Grid Search")

# Load the dataset with the correct delimiter
file_path = "C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\all_features.csv"
data = pd.read_csv(file_path, sep='\t', header=None)

# Manually assign column names
# Assuming the first column is the target (image type) and the rest are features
column_names = ['image_type'] + [f'feature{i}' for i in range(1, data.shape[1])]
data.columns = column_names

# Verify the dataset shape and class distribution
print("Dataset shape:", data.shape)
print("First few rows of the dataset:\n", data.head())

# Define features and target
target = 'image_type'  # The first column is the target (image type)
features = column_names[1:]  # All other columns are features
X = data[features]
y = data[target]

# Check class distribution to ensure all 100 rows per class are included
class_counts = y.value_counts()
print("Class distribution:\n", class_counts)

# Ensure every class has exactly 100 rows
if not all(class_counts == 100):
    raise ValueError("Some classes do not have exactly 100 rows. Please check the dataset.")

# Define hyperparameter grid
tree_numbers = range(25, 376, 50)  # Nt: Number of trees
predictor_numbers = [2, 4, 6, 8]  # Np: Number of predictors at each node

# Initialize variables to store the best results
best_accuracy = 0
best_params = (0, 0)  # (Nt, Np)
results = []

# Perform grid search
for Nt in tree_numbers:
    for Np in predictor_numbers:
        # Initialize the random forest classifier
        rf = RandomForestClassifier(n_estimators=Nt, max_features=Np, random_state=42)
        
        # Perform 5-fold cross-validation
        scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        mean_accuracy = scores.mean()
        
        # Store the results
        results.append((Nt, Np, mean_accuracy))
        
        # Update the best parameters if the current model is better
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = (Nt, Np)

# Convert results to a DataFrame for visualization
results_df = pd.DataFrame(results, columns=['Nt', 'Np', 'Accuracy'])

# Print the best parameters and accuracy
print(f"Best Parameters: Nt = {best_params[0]}, Np = {best_params[1]}")
print(f"Best Cross-Validated Accuracy: {best_accuracy:.4f}")

# Visualize the results
pivot_table = results_df.pivot(index='Np', columns='Nt', values='Accuracy')
plt.figure(figsize=(10, 6))
plt.title('Cross-Validated Accuracy for Random Forest Models')
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.xlabel('Number of Trees (Nt)')
plt.ylabel('Number of Predictors (Np)')
plt.show()

# Interpretation
print("\nInterpretation:")
print("1. The heatmap shows the cross-validated accuracy for different combinations of Nt and Np.")
print("2. The best combination of Nt and Np achieves the highest accuracy, balancing the number of trees and predictors.")
print("3. Increasing Nt generally improves accuracy, but the effect diminishes after a certain point.")
print("4. The optimal Np depends on the dataset; too few predictors may underfit, while too many may overfit.")

# Section 3.2: Variability of Accuracy Across Independent Runs
print("\nSection 3.2: Variability of Accuracy Across Independent Runs")

# Best parameters from Section 3.1
best_Nt, best_Np = best_params

# Refit the model 15 times and collect cross-validated accuracies
accuracies = []
for i in range(15):
    rf = RandomForestClassifier(n_estimators=best_Nt, max_features=best_Np, random_state=i)
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    accuracies.append(scores.mean())

# Calculate mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

# Perform a one-sample t-test to check if the model performs better than chance
chance_level = 1 / len(class_counts)  # Assuming uniform random guessing
t_stat, p_value = ttest_1samp(accuracies, chance_level)
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpretation
print("\nInterpretation:")
if p_value < 0.05:
    print("The model performs significantly better than chance (p < 0.05).")
else:
    print("The model does not perform significantly better than chance (p >= 0.05).")
print("The mean accuracy and standard deviation provide insight into the variability of the model's performance across independent runs.")