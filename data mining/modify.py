import pandas as pd
import numpy as np
# Load the CSV file
input_file = "train.csv"  # Replace with your file path
output_file = "newtrain.csv"

# Read the CSV into a DataFrame
df = pd.read_csv(input_file)

# Function to categorize age
def categorize_age(age):
    if age < 13:
        return 1  # Kid
    elif 13 <= age <= 19:
        return 2  # Teen
    elif 20 <= age <= 35:
        return 3  # 20 to 35
    elif 36 <= age <= 50:
        return 4  # 35 to 50
    elif 51 <= age <= 70:
        return 5  # 50 to 70
    else:
        return 6  # 70+

# Apply categorization to the Age column
df['Age'] = df['Age'].apply(categorize_age)

# Example: Modify other columns (add logic as needed)
# For Vehicle_Age: Map 1->New, 2->1-2 years, 3->>2 years

def group_regions(code):
    if code <= 10:
        return 1
    elif 11 <= code <= 20:
        return 2
    elif 21 <= code <= 30:
        return 3
    else:
        return 4

df['Region_Code'] = df['Region_Code'].apply(group_regions)
df['Annual_Premium'] = np.log1p(df['Annual_Premium']).astype(int)  # log1p avoids log(0)
df['Vintage'] = (df['Vintage'] / 30).astype(int)

# Save the modified DataFrame back to a CSV
df.to_csv(output_file, index=False)

print(f"Modified data saved to {output_file}")