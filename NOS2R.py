from sklearn.preprocessing import StandardScaler

# Load your dataset into a DataFrame
df = pd.read_csv('original_data.csv')

# Initialize the scaler
scaler = StandardScaler()

# Apply the scaler to the relevant columns
normalized_data = scaler.fit_transform(df)

# Convert the normalized data back into a DataFrame
df_normalized = pd.DataFrame(normalized_data, columns=df.columns)

# Check the result
print(df_normalized.head())



import numpy as np

# Create a 22x22 scaling matrix (diagonal matrix)
Scale = np.eye(22)
Scale[0, 0] = 1  # Scale first dimension by 1
Scale[1, 1] = 2  # Scale second dimension by 2
Scale[2, 2] = 3  # Scale third dimension by 3

# Apply the scaling matrix
scaled_data = df_normalized @ Scale

# Create 22x22 shearing matrices by embedding 3x3 matrices
Shx = np.eye(22)
Shx[0, 1] = 1.5  # Shear factor for x
Shx[0, 2] = 2.5  # Shear factor for x

Shy = np.eye(22)
Shy[1, 0] = 1.5  # Shear factor for y
Shy[1, 2] = 2.5  # Shear factor for y

Shz = np.eye(22)
Shz[2, 0] = 1.5  # Shear factor for z
Shz[2, 1] = 2.5  # Shear factor for z

# Applying shearing transformations
Sheared_datax = scaled_data @ Shx
Sheared_datay = Sheared_datax @ Shy
Sheared_data = Sheared_datay @ Shz

# Create 22x22 reflection matrices
Rfx = np.eye(22)
Rfx[0, 0] = -1  # Reflection across x-axis

Rfy = np.eye(22)
Rfy[1, 1] = -1  # Reflection across y-axis

Rfz = np.eye(22)
Rfz[2, 2] = -1  # Reflection across z-axis

# Applying reflection transformations
Reflected_data1 = Sheared_data @ Rfx
Reflected_data2 = Reflected_data1 @ Rfy
Reflected_data = Reflected_data2 @ Rfz

# Final result
NOS2R_data = Reflected_data

print(NOS2R_data)


# Save to CSV
NOS2R_data.to_csv('NOS2R_data.csv', index=False)