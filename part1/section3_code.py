import pandas as pd
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import seaborn as sns

# References used 
# 3.3: https://stackoverflow.com/questions/588004/is-floating-point-math-broken this was to understand to use a tolerance for the p-values
# 3.4: https://www.geeksforgeeks.org/python-pandas-dataframe-corr/ this was to understand how to calculate the correlation matrix

# this removes that toolbar at the bottom of the plot
plt.rcParams['toolbar'] = 'None'

# this is the path to the CSV file being read in
file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\40402274_features.csv'
data = pd.read_csv(file_path)

# this defines the labels for the letters and non-letters
letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'}
non_letters = {'sad', 'smiley', 'xclaim'}

# this filters the data into letters and non-letters
letters_data = data[data['LABEL'].isin(letters)]
non_letters_data = data[data['LABEL'].isin(non_letters)]

# this stores the 6 features for the first part
first_part_features = ['nr_pix', 'rows_with_1', 'cols_with_1', 'rows_with_3p', 'cols_with_3p', 'aspect_ratio']

# this is the list of all features
all_features = [
    'nr_pix', 'rows_with_1', 'cols_with_1', 'rows_with_3p', 'cols_with_3p', 'aspect_ratio',
    'neigh_1', 'no_neigh_above', 'no_neigh_below', 'no_neigh_left', 'no_neigh_right',
    'no_neigh_horiz', 'no_neigh_vert', 'connected_areas', 'eyes', 'centroid_spread'
]

# this just initialises the index for the first histogram
current_index = 0

# this will be used for when the user wants to navigate between parts of the program
enter_pressed_first_part = False 
enter_pressed_second_part = False
enter_pressed_third_part = False


# 3.1 -----------------------------------------------------------------------------------------------------------------------
print("\nProceeding to the first part (3.1)...\n")
def update_histogram():
    plt.clf()  
    #this gets the feature and plots the histogram
    feature = first_part_features[current_index]
    plt.hist(data[feature], bins=10, color='blue', alpha=0.7)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    
    # this tells the user how to navigate between the histograms
    plt.text(0.5, 1.10, "This is Question 3.1\nPress ← or → to navigate between histograms\nPress Enter to proceed to 3.2", 
             ha='center', va='center', transform=ax.transAxes, fontsize=12, color='red')
    plt.draw()

# this handles the key press events 
def on_key_first_part(event):
    global current_index, enter_pressed_first_part
    if event.key == 'right':
        current_index = (current_index + 1) % len(first_part_features)  
    elif event.key == 'left':
        current_index = (current_index - 1) % len(first_part_features)  
    elif event.key == 'enter':
        enter_pressed_first_part = True
        plt.close()  
        return
    update_histogram()

# this handles when the user closes the window
def on_close(event):
    if not enter_pressed_first_part and not enter_pressed_second_part:
        print("Window closed. Exiting program.")
        sys.exit()  # Terminate the program

# this creates the initial plot and calls the update function
fig, ax = plt.subplots(figsize=(10, 6))
update_histogram()

# this adds the key press and close-window events to the handler
fig.canvas.mpl_connect('key_press_event', on_key_first_part)
fig.canvas.mpl_connect('close_event', on_close)

# this shows the plot
plt.show()

# 3.2 -----------------------------------------------------------------------------------------------------------------------
# when the user presses enter in the first part
if enter_pressed_first_part:
    print("\nProceeding to the second part (3.2)...\n")

    # this calculates the summary statistics 
    def calculate_statistics(df, features):
        stats = {}
        for feature in features:
            stats[feature] = {
                'mean': df[feature].mean(),
                'median': df[feature].median(),
                'std': df[feature].std(),
                'min': df[feature].min(),  
                'max': df[feature].max()   
            }
        return pd.DataFrame(stats)

    # this calls the calculate function for both letters and non-letters
    letters_stats = calculate_statistics(letters_data, all_features)
    non_letters_stats = calculate_statistics(non_letters_data, all_features)

    # this just splits the features into 2 groups so it will fit in the console output
    features_group1 = all_features[:8]  
    features_group2 = all_features[8:]  

    # this prints the statstics for the letters
    print("Summary Statistics for Letters (Part 1/2):")
    print(letters_stats[features_group1]) # first group
    print("\nSummary Statistics for Letters (Part 2/2):")
    print(letters_stats[features_group2]) # second group

    # this prints the statistics for the non-letters
    print("\nSummary Statistics for Non-Letters (Part 1/2):")
    print(non_letters_stats[features_group1]) # first group
    print("\nSummary Statistics for Non-Letters (Part 2/2):")
    print(non_letters_stats[features_group2]) # second group

    # this stores the three features I find interesting and want to display 
    interesting_features = ['nr_pix', 'connected_areas', 'aspect_ratio']

    # this initialises the index for the second histogram
    current_index_second_part = 0

    
    def update_histogram_second_part():
        plt.clf()  # this clears the histogram

        # this gets the feature and plots the histogram
        feature = interesting_features[current_index_second_part]
        plt.hist(letters_data[feature], bins=10, alpha=0.5, label='Letters', color='blue')
        plt.hist(non_letters_data[feature], bins=10, alpha=0.5, label='Non-Letters', color='red')
        plt.title(f'Distribution of {feature} for Letters and Non-Letters')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.legend()
        
        # this tells the user how to navigate between the histograms
        plt.text(0.5, 1.10, "This is Question 3.2\nPress ← or → to navigate between histograms\nPress Enter to proceed to 3.3", 
                 ha='center', va='center', transform=ax_second_part.transAxes, fontsize=12, color='red')
        plt.draw()

    # this handles the key press events for the second part
    def on_key_second_part(event):
        global current_index_second_part, enter_pressed_second_part
        if event.key == 'right':
            current_index_second_part = (current_index_second_part + 1) % len(interesting_features)  
        elif event.key == 'left':
            current_index_second_part = (current_index_second_part - 1) % len(interesting_features)  
        elif event.key == 'enter':
            enter_pressed_second_part = True
            plt.close()  
            return
        update_histogram_second_part()

    # this creates the initial plot and calls the update function
    fig_second_part, ax_second_part = plt.subplots(figsize=(10, 6))
    update_histogram_second_part()

    # this adds the key press and close-window events to the handler
    fig_second_part.canvas.mpl_connect('key_press_event', on_key_second_part)
    fig_second_part.canvas.mpl_connect('close_event', on_close)

    plt.show()


# 3.3 -----------------------------------------------------------------------------------------------------------------------
# when the user presses enter in the second part
if enter_pressed_second_part:
    print("\nProceeding to the third part (3.3)...\n")

    # this calculates the p-values for each feature
    p_values = {}
    for feature in all_features:
        t_stat, p_value = stats.ttest_ind(letters_data[feature], non_letters_data[feature])
        p_values[feature] = p_value

    # this sorts the features by p-value
    sorted_features = sorted(p_values.keys(), key=lambda x: p_values[x])

    # this prints the values
    print("P-values for each feature (up to 20 decimal points):")
    for feature in sorted_features:
        print(f"{feature}: {p_values[feature]:.20f}")

    # this finds the most discriminatory features
    lowest_p_value = min(p_values.values())
    tolerance = 1e-10  # this is the tolerance for the p-values
    most_discriminatory_features = [feature for feature, p in p_values.items() if abs(p - lowest_p_value) < tolerance]

    # this prints the most discriminatory features
    print("\nMost discriminatory features:")
    for feature in most_discriminatory_features:
        print(f"{feature}: {p_values[feature]:.20f}")

    # this initialises the index for the third histogram
    current_index_third_part = 0

    def update_histogram_third_part():
        plt.clf()  # this clears the histogram
        feature = most_discriminatory_features[current_index_third_part]
        
        # this checks if the feature is diverse or not
        if feature in ['connected_areas', 'eyes']:  
            # this uses a bar plot for non-diverse
            plt.bar(['Letters', 'Non-Letters'], 
                    [letters_data[feature].mean(), non_letters_data[feature].mean()], 
                    color=['blue', 'red'], alpha=0.7)
            plt.title(f'Mean {feature} for Letters and Non-Letters')
            plt.ylabel(f'Mean {feature}')

        # this uses a boxplot for diverse features
        else:
            plt.boxplot([letters_data[feature], non_letters_data[feature]], tick_labels=['Letters', 'Non-Letters'], notch=True, whis=1.5)
            plt.title(f'Distribution of {feature}')
            plt.ylabel(feature)
        
        # this tells the user how to navigate between the histograms
        plt.text(0.5, 1.15, "Section 3.3\nPress ← or → to navigate between histograms\nPress Enter to proceed to 3.4", 
                 ha='center', va='center', transform=ax_third_part.transAxes, fontsize=12, color='red')
        plt.draw()

    # this handles the key press events for the third part
    def on_key_third_part(event):
        global current_index_third_part, enter_pressed_third_part
        if event.key == 'right':
            current_index_third_part = (current_index_third_part + 1) % len(most_discriminatory_features)  
        elif event.key == 'left':
            current_index_third_part = (current_index_third_part - 1) % len(most_discriminatory_features)  
        elif event.key == 'enter':
            enter_pressed_third_part = True
            plt.close()  
            return
        update_histogram_third_part()

    # this creates the initial plot and calls the update function
    fig_third_part, ax_third_part = plt.subplots(figsize=(10, 6))
    update_histogram_third_part()

    # this adds the key press and close-window events to the handler
    fig_third_part.canvas.mpl_connect('key_press_event', on_key_third_part)
    fig_third_part.canvas.mpl_connect('close_event', on_close)

    plt.show()

# 3.4 -----------------------------------------------------------------------------------------------------------------------
# when the user presses enter in the third part
if enter_pressed_third_part:
    print("Proceeding to the fourth part (3.4)...")

    # this calculates the correlation matrix using pearon correlation
    correlation_matrix = data[all_features].corr()

    # this finds the highly correlated feature pairs
    high_correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.85:
                high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

    # this sorts the high_correlation_pairs by correlation value
    high_correlation_pairs = sorted(high_correlation_pairs, key=lambda x: abs(x[2]), reverse=True)

    # this prints the highly correlated feature pairs
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.85):")
    count = 1
    for pair in high_correlation_pairs:
        # this prints the feature pairs and their correlation values as well as numbers each of them 
        print(f"{count}. [{pair[0]}, {pair[1]}]: {pair[2]:.4f}")
        count += 1

    # this plots the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Matrix of Features")
    plt.show()

    print("\n\nEnd of the program.\n")