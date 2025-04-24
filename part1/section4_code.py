import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# References used 
# 4.1: https://www.geeksforgeeks.org/ml-multiple-linear-regression-backward-elimination-technique/ this ws used to understand the backward elimination technique
# 4.2: https://www.datacamp.com/tutorial/understanding-logistic-regression-python this was used to understand logistic regression

# this is the path to the CSV file being read in
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\40402274_features.csv'
data = pd.read_csv(file_path)

# this defines the labels for the letters and non-letters
letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
non_letters = {'sad', 'smiley', 'xclaim'}

# this creates a new column to indicate if the image is a letter
data['is_letter'] = data['LABEL'].apply(lambda x: 1 if x in letters else 0)

# 4.1 -----------------------------------------------------------------------------------------------------------------------
print("\nProceeding to first part (4.1)\n")

# this is the list of features to be used in the regression
features = [
    'nr_pix', 'rows_with_1', 'cols_with_1', 'rows_with_3p', 'cols_with_3p',
    'neigh_1', 'no_neigh_above', 'no_neigh_below', 'no_neigh_left', 'no_neigh_right',
    'no_neigh_horiz', 'no_neigh_vert', 'connected_areas', 'eyes', 'centroid_spread'
]

# this prepares the data for regression
# X is the independent variables and y is the dependent variable
X = data[features]
y = data['aspect_ratio']

# this adds a constant to the independent variables
X = sm.add_constant(X)

# this implements the backward elimination technique to select the significant features
def backward_elimination(X, y, significance_level=0.05):
    features = list(X.columns)
    while len(features) > 0:
        X_subset = X[features]
        X_subset = sm.add_constant(X_subset)  
        model = sm.OLS(y, X_subset).fit()
        p_values = model.pvalues[1:]  
        max_p_value = p_values.max()
        if max_p_value > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
        else:
            break
    return features

# this calls the backward elimination function 
significant_features = backward_elimination(X, y)
print("\nSignificant features after backward elimination:\n", significant_features, "\n")

# this fits the parsimonious model using the significant features
X_parsimonious = X[significant_features]
X_parsimonious = sm.add_constant(X_parsimonious)  
parsimonious_model = sm.OLS(y, X_parsimonious).fit()

# this prints the summary of the parsimonious model
print(parsimonious_model.summary())


# 4.2 -----------------------------------------------------------------------------------------------------------------------
print("\nProceeding to second part (4.2)\n")

# this sets nr_pix as the most useful feature
most_useful_feature = 'nr_pix'

# this prepares the data for logistic regression
X_logistic = data[[most_useful_feature]]  
y_logistic = data['is_letter']

# this creates the logistic regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_logistic, y_logistic)

# this prints the results of the logistic regression
print("Logistic Regression Results:")
print(f"Coefficient for {most_useful_feature}: {logistic_model.coef_[0][0]:.4f}")
print(f"Intercept: {logistic_model.intercept_[0]:.4f}\n")

# this predicts the probability of being a letter for each value of nr_pix
X_test = np.linspace(X_logistic.min(), X_logistic.max(), 300).reshape(-1, 1)
X_test_df = pd.DataFrame(X_test, columns=[most_useful_feature])  
y_prob = logistic_model.predict_proba(X_test_df)[:, 1]

# this hides the toolbar in the plot
plt.rcParams['toolbar'] = 'None' # this hides the toolbar 

# this plots the logistic regression curve
plt.figure(figsize=(10, 6))
plt.scatter(X_logistic, y_logistic, color='blue', label='Data Points', alpha=0.6)
plt.plot(X_test, y_prob, color='red', label='Logistic Regression Curve')
plt.axhline(0.5, color='black', linestyle='--', label='Decision Boundary (p = 0.5)')  
plt.title(f'Logistic Regression for {most_useful_feature}\n(Probability of Being a Letter)')
plt.xlabel(f'{most_useful_feature} (Number of Black Pixels)')
plt.ylabel('Probability of Being a Letter')
plt.legend()
plt.show()


# 4.3 -----------------------------------------------------------------------------------------------------------------------
print("\nProceeding to third part (4.3)\n")

# this defines the features to base the median splits
features_to_split = ['nr_pix', 'aspect_ratio', 'neigh_1']

# this creates new categorical features based on the median splits
for feature in features_to_split:
    median_value = data[feature].median()
    data[f'split_{feature}'] = (data[feature] > median_value).astype(int)

# this defines the classes for the new categorical features
classes = {
    'Letters': data['LABEL'].isin(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
    'Faces': data['LABEL'].isin(['smiley', 'sad']),
    'Exclamation Marks': data['LABEL'] == 'xclaim'
}

# this calculates the proportion of "1"s for each class and feature
proportions = {f'split_{feature}': {} for feature in features_to_split}

# this calculates the proportion of "1"s for each class and feature
for feature in features_to_split:
    split_feature = f'split_{feature}'
    for class_name, mask in classes.items():
        proportions[split_feature][class_name] = data.loc[mask, split_feature].mean()

# this prints the proportions of "1"s for each class and feature
proportions_df = pd.DataFrame(proportions)
print("\nProportion of '1's for Each Class:")
print(proportions_df)

# -----------------------------------------------------------------------------------------------------------------------
print("\n\nEnd of Section 4.\n")