import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp

#------------------------------------------------------------------------------------------------------------------------#
# Section 3.1: Random Forest Grid Search
print("\nSection 3.1: Random Forest Grid Search")

# loads the dataset
file_path = "C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\Assignment2\\all_features.csv"
data = pd.read_csv(file_path, sep='\t', header=None)

# checks if the dataset has 100 rows for each class
column_names = ['image_type'] + [f'feature{i}' for i in range(1, data.shape[1])]
data.columns = column_names

# verifies the dataset shape and first few rows
print("Dataset shape:", data.shape)
print("First few rows of the dataset:\n", data.head())

# defines the target and features
target = 'image_type'  
features = column_names[1:] 
X = data[features]
y = data[target]

# checks the class distribution
class_counts = y.value_counts()
print("Class distribution:\n", class_counts)

# checks if all classes have exactly 100 rows
if not all(class_counts == 100):
    raise ValueError("Some classes do not have exactly 100 rows. Please check the dataset.")

# set the hyperparameters for the grid search
tree_numbers = range(25, 376, 50)  
predictor_numbers = [2, 4, 6, 8]  

# initialises variables to store the best parameters and accuracy
best_accuracy = 0
best_params = (0, 0) 
results = []

# performs grid search for the best parameters
for Nt in tree_numbers:
    for Np in predictor_numbers:
        # initialises the Random Forest classifier with the current parameters
        rf = RandomForestClassifier(n_estimators=Nt, max_features=Np, random_state=42)
        
        # performs 5-fold cross-validation and calculates the mean accuracy
        scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        mean_accuracy = scores.mean()
        
        # stores the results
        results.append((Nt, Np, mean_accuracy))
        
        # updates the best parameters if the current accuracy is better
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_params = (Nt, Np)

# creates a DataFrame to store the results
results_df = pd.DataFrame(results, columns=['Nt', 'Np', 'Accuracy'])

# Print the best parameters and accuracy
print(f"\n\nBest Parameters: Nt = {best_params[0]}, Np = {best_params[1]}")
print(f"Best Cross-Validated Accuracy: {best_accuracy:.4f}")

# visualises the results using a heatmap
pivot_table = results_df.pivot(index='Np', columns='Nt', values='Accuracy')
plt.figure(figsize=(10, 6))
plt.title('Cross-Validated Accuracy for Random Forest Models')
sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap='viridis', cbar_kws={'label': 'Accuracy'})
plt.xlabel('Number of Trees (Nt)')
plt.ylabel('Number of Predictors (Np)')
plt.show()



#------------------------------------------------------------------------------------------------------------------------#

# Section 3.2: Variability of Accuracy Across Independent Runs
print("\n\n\nSection 3.2: Variability of Accuracy Across Independent Runs\n")

# refit the model with the best parameters
best_Nt, best_Np = best_params

# initialises a list to store accuracies
accuracies = []
# performs 15 independent runs with different random states
for i in range(15):
    rf = RandomForestClassifier(n_estimators=best_Nt, max_features=best_Np, random_state=i)
    scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    accuracies.append(scores.mean())

# calculates the mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

# performs a t-test to check if the mean accuracy is significantly different from chance level
chance_level = 1 / len(class_counts) 
t_stat, p_value = ttest_1samp(accuracies, chance_level)
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}\n")

