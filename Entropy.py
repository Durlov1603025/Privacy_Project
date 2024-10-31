import numpy as np
import pandas as pd

def calculate_entropy(data):
    # Flatten the data to a 1D array
    flattened_data = data.values.flatten()
    
    # Calculate value counts and probabilities
    value_counts = pd.Series(flattened_data).value_counts()
    probabilities = value_counts / value_counts.sum()
    
    # Calculate Shannon entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)
    return entropy



#initialization
orgData = pd.read_csv('original_data.csv')
NOS2R_data = pd.read_csv('NOS2R_data.csv') 
NOS2R2_data = pd.read_csv('NOS2R2_data.csv') 

# Calculate entropy for the Original dataset
org_entropy = calculate_entropy(orgData_data)
print(f"Entropy of Original dataset: {org_entropy}")


# Calculate entropy for the NOS2R data
NOS2R_entropy = calculate_entropy(NOS2R_data)
print(f"Entropy of NOS2R Data: {NOS2R_entropy}")

# Calculate entropy for the NOS2R2 data
NOS2R2_entropy = calculate_entropy(NOS2R2_data)
print(f"Entropy of NOS2R2 Data: {NOS2R2_entropy}")
