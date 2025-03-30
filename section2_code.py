import os
import csv
import numpy as np  
from scipy.ndimage import label, generate_binary_structure

# Reference for SciPy's label function:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html

def count_eyes(image_matrix):

    # this defines the structure for orthoganl connections
    structure_4 = generate_binary_structure(2, 1)

    # this labels the whitespace connected regions
    labeled_whitespace, num_whitespace_regions = label(image_matrix == 1, structure=structure_4)

    # this initializes the number of eyes to 0
    num_eyes = 0

    # this loops through all the whitespace regions
    for region_id in range(1, num_whitespace_regions + 1):
        # this creates a mask for the current region
        region_mask = (labeled_whitespace == region_id)
        
        # this creates a mask for the border pixels of the region
        border_pixels = np.logical_xor(region_mask, np.pad(region_mask[1:-1, 1:-1], 1, mode='constant', constant_values=0))
        
        # this checks if all the border pixels are black
        if np.all(image_matrix[border_pixels] == 0):
            num_eyes += 1

    return num_eyes

# this function calculates the rest of the 16 features
def calculate_features(image_matrix):

    # this calculates the number of black pixels in the image
    nr_pix = np.sum(image_matrix == 0)

    # this calculates the number of rows with only one black pixel
    rows_with_1 = np.sum(np.sum(image_matrix == 0, axis=1) == 1)

    # this calculates the number of columns with only one black pixel
    cols_with_1 = np.sum(np.sum(image_matrix == 0, axis=0) == 1)

    # this calculates the number of rows with 3 or more
    rows_with_3p = np.sum(np.sum(image_matrix == 0, axis=1) >= 3)

    # this calculates the number of columns with 3 or more
    cols_with_3p = np.sum(np.sum(image_matrix == 0, axis=0) >= 3)
    
    # this calculates the aspect ratio of the image (width/height)
    black_pixels = np.argwhere(image_matrix == 0)
    if len(black_pixels) == 0:
        aspect_ratio = 1.0
    else:
        height = black_pixels[:, 0].max() - black_pixels[:, 0].min() + 1
        width = black_pixels[:, 1].max() - black_pixels[:, 1].min() + 1
        aspect_ratio = width / height if height != 0 else 1.0
    
    # this calculates the number of black pixels with exactly one black neighbour
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    padded_image = np.pad(image_matrix == 0, 1, mode='constant', constant_values=0)
    neighbour_counts = np.zeros_like(image_matrix, dtype=int)
    for i in range(1, 19):
        for j in range(1, 19):
            neighbour_counts[i-1, j-1] = np.sum(padded_image[i-1:i+2, j-1:j+2] * kernel)
    
    neight_1 = np.sum((image_matrix == 0) & (neighbour_counts == 1))

    # this calculates the number of black pixels with no black neighbour above
    no_neigh_above = np.sum((image_matrix == 0) &
                             (np.roll(image_matrix, 1, axis=0) != 0) &  
                             (np.roll(image_matrix, (1, 1), axis=(0, 1)) != 0) &  
                             (np.roll(image_matrix, (1, -1), axis=(0, 1)) != 0))  
    # this calculates the number of black pixels with no black neighbour below
    no_neigh_below = np.sum((image_matrix == 0) &
                             (np.roll(image_matrix, -1, axis=0) != 0) &  
                             (np.roll(image_matrix, (-1, 1), axis=(0, 1)) != 0) &  
                             (np.roll(image_matrix, (-1, -1), axis=(0, 1)) != 0))  
    # this calculates the number of black pixels with no black neighbour to the left
    no_neigh_left = np.sum((image_matrix == 0) &
                            (np.roll(image_matrix, 1, axis=1) != 0) &  
                            (np.roll(image_matrix, (1, 1), axis=(0, 1)) != 0) &  
                            (np.roll(image_matrix, (-1, 1), axis=(0, 1)) != 0))  
    # this calculates the number of black pixels with no black neighbour to the right
    no_neigh_right = np.sum((image_matrix == 0) &
                             (np.roll(image_matrix, -1, axis=1) != 0) &  
                             (np.roll(image_matrix, (1, -1), axis=(0, 1)) != 0) &  
                             (np.roll(image_matrix, (-1, -1), axis=(0, 1)) != 0))  
    # this calculates the number of black pixels with no black neighbour horizontally
    no_neigh_horiz = np.sum((image_matrix == 0) &
                             (np.roll(image_matrix, 1, axis=1) != 0) &  
                             (np.roll(image_matrix, -1, axis=1) != 0))  
    # this calculates the number of black pixels with no black neighbour vertically
    no_neigh_vert = np.sum((image_matrix == 0) &
                            (np.roll(image_matrix, 1, axis=0) != 0) &  
                            (np.roll(image_matrix, -1, axis=0) != 0))  
    
    # this calculates the number of connected areas in the image
    structure = generate_binary_structure(2, 2)
    _, connected_areas = label(image_matrix == 0, structure=structure)
    
    # calls the eyes function to calculate the number of eyes
    num_eyes = count_eyes(image_matrix)
    
    # this calculates the centroid spread of the black pixels
    center = np.array([8.5, 8.5])
    distances = np.linalg.norm(black_pixels - center, axis=1) if len(black_pixels) > 0 else np.array([0])
    centroid_spread = np.sum(distances)
    
    # returns all the 16 calculated features
    return [
        nr_pix, rows_with_1, cols_with_1, rows_with_3p, cols_with_3p, aspect_ratio,
        neight_1, no_neigh_above, no_neigh_below, no_neigh_left, no_neigh_right,
        no_neigh_horiz, no_neigh_vert, connected_areas, num_eyes, centroid_spread
    ]

# this function processes all the csv files in the folder
def process_csv_files(csv_folder, output_file):

    # this is the header of the output CSV file
    header = [
        "LABEL", "INDEX", "nr_pix", "rows_with_1", "cols_with_1", "rows_with_3p", "cols_with_3p",
        "aspect_ratio", "neigh_1", "no_neigh_above", "no_neigh_below", "no_neigh_left", "no_neigh_right",
        "no_neigh_horiz", "no_neigh_vert", "connected_areas", "eyes", "centroid_spread"
    ]
    rows = []

    # this loops through all the files in the folder and processes them
    for filename in sorted(os.listdir(csv_folder)):
        if filename.endswith(".csv"):
            try:
                parts = filename.split("_")
                if len(parts) == 3:
                    label = parts[1]
                    index = int(parts[2].split(".")[0])
                    csv_path = os.path.join(csv_folder, filename)
                    with open(csv_path, 'r') as file:
                        reader = csv.reader(file)
                        image_matrix = np.array([list(map(int, row)) for row in reader])
                    features = calculate_features(image_matrix)
                    rows.append([label, index] + features)
                else:
                    print(f"Skipping file {filename}: Invalid filename format.")
            except (ValueError, IndexError) as e:
                print(f"Skipping file {filename}: Error processing file. Details: {e}")

    rows.sort(key=lambda x: (x[0], x[1]))
    
    # this writes the features to the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)


# this is the main code that processes the CSV files
csv_folder = "C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\images"
output_file = "C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\40402274_features.csv"
process_csv_files(csv_folder, output_file)
print(f"Features saved to {output_file}")
