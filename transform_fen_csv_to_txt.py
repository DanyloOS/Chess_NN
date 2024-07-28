import pandas as pd
import sys

# Load the CSV file
# csv_file_path = 'path/to/your/file.csv'  # Replace with your CSV file path
csv_file_path = input()
df = pd.read_csv(csv_file_path, sep=';')

# Extract the FEN column
fen_data = df['FEN']

# Save the FEN data to a text file
# output_file_path = 'path/to/your/output.txt'  # Replace with your desired output file path
output_file_path = f'{csv_file_path}.txt'  # Replace with your desired output file path
with open(output_file_path, 'w') as f:
    for fen in fen_data:
        f.write(f"{fen}\n")

print(f"FEN data has been saved to {output_file_path}.")