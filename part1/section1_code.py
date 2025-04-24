import csv

# Function to convert a PGM file to a CSV file

def pgm_to_csv(pgm_file_path, csv_file_path):
    with open(pgm_file_path, 'r') as pgm_file:
        # Reads the header of the PGM file
        magic_number = pgm_file.readline().strip()
        if magic_number != 'P2':
            raise ValueError("File is not a valid PGM (P2) file.")

        # skips all of the comments in the PGM file
        line = pgm_file.readline().strip()
        while line.startswith('#'):
            line = pgm_file.readline().strip()

        # reads the width and height of the image from the PGM file
        width, height = map(int, line.split())
        max_value = int(pgm_file.readline().strip())

        # reads the pixel values from the PGM file
        pixels = []
        for line in pgm_file:
            pixels.extend(map(int, line.split()))

    # writes the pixel values to the CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for y in range(height):
            row = pixels[y * width:(y + 1) * width]
            writer.writerow(row)

# main function to convert the PGM file to a CSV file
if __name__ == "__main__":
    pgm_file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\PGM Files\\xclaim\\40402274_xclaim_20.pgm'  # My file path for the PGM file I want to convert 
    csv_file_path = 'C:\\Users\\JoshZ\\OneDrive\\University\\CSC2062\\images\\xclaim\\40402274_xclaim_20.csv'  # The path of the created CSV file
    pgm_to_csv(pgm_file_path, csv_file_path)
    print(f"PGM file converted successfully")