# Rotation functions for xy, yz, and xz planes
def r_xy(theta):
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [np.sin(theta)*np.sin(theta), np.cos(theta), np.sin(theta)*np.cos(theta)],
                     [np.sin(theta)*np.cos(theta), -np.sin(theta), np.cos(theta)*np.cos(theta)]])

def r_yz(theta):
    return np.array([[np.cos(theta)**2, -np.sin(theta)*np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta), 0],
                     [np.sin(theta)*np.cos(theta), -np.sin(theta)**2, np.cos(theta)]])

def r_xz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta)*np.cos(theta), np.cos(theta)**2, np.sin(theta)],
                     [-np.sin(theta)**2, np.sin(theta)*np.cos(theta), np.cos(theta)]])

# Store all rotation functions in a list
rotation_functions = [r_xy, r_yz, r_xz]


# Create triplets from columns
def create_triplets(column_names):
    triplets = [column_names[i:i+3] for i in range(0, len(column_names), 3)]
    remainder = len(column_names) % 3
    if remainder:
        last_triplet = triplets[-1]
        for i in range(3 - remainder):
            last_triplet.append(column_names[-(remainder + i + 1)])
    return triplets



column_data = NOS2R_data.columns.to_list()
triplets = create_triplets(column_data)
angles = list(range(1, 361))                     # Take angles from 0 to 360 degrees




def rotate_all_angles(dataset, rotation_angles, rotation_func, triplets):
    rotated_data = []

    for angle in rotation_angles:
        theta = np.radians(angle)
        rotated_dataset = dataset.copy()

        for triplet in triplets:
            triplet_cols = [f"{col}_rot_angle_{angle}" for col in triplet]
            for idx in range(len(rotated_dataset)):
                original_values = rotated_dataset.loc[idx, triplet].to_numpy().reshape(3, 1)

                # Apply the rotation matrix
                rotated_values = np.dot(rotation_func(theta), original_values).flatten()

                # Assign rotated values to new columns
                rotated_dataset.loc[idx, triplet_cols] = rotated_values

        rotated_dataset['angle'] = angle  # Add the rotation angle to the dataset
        rotated_data.append(rotated_dataset)  # Append each rotated dataset

    return pd.concat(rotated_data, ignore_index=True)





rotated_xy_data = rotate_all_angles(NOS2R_data, angles, r_xy, triplets)
rotated_yz_data = rotate_all_angles(NOS2R_data, angles, r_yz, triplets)
rotated_xz_data = rotate_all_angles(NOS2R_data, angles, r_xz, triplets)



# Function to calculate variance of differences between the original and rotated datasets for each angle and plane
def calculate_variances(original_data, rotated_data, columns):
    variance_data = {}

    # Calculate variance for each column
    for col_index, col_name in enumerate(columns):
        variance_data[col_name] = np.var(original_data[:, col_index] - rotated_data[:, col_index])

    return variance_data





# Calculate variances for all angles and planes after creating the rotated data
def calculate_all_variances(original_data, rotated_xy_data, rotated_yz_data, rotated_xz_data, columns):
    # Initialize lists to hold variances for each plane and angle
    variances_xy = []
    variances_yz = []
    variances_xz = []

    for angle in range(1, 361):
        # Select rotated data for each angle
        rotated_data_xy = rotated_xy_data[angle - 1::360, :]  # Rows corresponding to this angle
        rotated_data_yz = rotated_yz_data[angle - 1::360, :]
        rotated_data_xz = rotated_xz_data[angle - 1::360, :]

        # Calculate variances for each plane at this angle
        variances_xy.append(calculate_variances(original_data, rotated_data_xy, columns))
        variances_yz.append(calculate_variances(original_data, rotated_data_yz, columns))
        variances_xz.append(calculate_variances(original_data, rotated_data_xz, columns))

    return variances_xy, variances_yz, variances_xz


variances_xy, variances_yz, variances_xz = calculate_all_variances(original_data_np, rotated_xy_data, rotated_yz_data, rotated_xz_data, column_data)



# Function to filter variances based on thresholds for each plane
def filter_variances_per_plane(df, thresholds):
    filtered_angles = []
    filtered_variances = {"angle": []}
    for col in df.columns:
        if col != "angle":
            filtered_variances[col] = []

    # Filter each row based on the thresholds
    for idx in range(len(df)):
        row = df.iloc[idx]
        angle = row["angle"]

        # Keep the row if all variances are within thresholds
        if all(row[col] <= thresholds[i] for i, col in enumerate(df.columns) if col != "angle"):
            filtered_angles.append(angle)
            filtered_variances["angle"].append(angle)
            for col in df.columns:
                if col != "angle":
                    filtered_variances[col].append(row[col])

    return pd.DataFrame(filtered_variances), filtered_angles



# Define the threshold for each variance column; customize as needed
thresholds_xy = [0.5] * (len(variances_xy.columns) - 1)
thresholds_yz = [0.2] * (len(variances_yz.columns) - 1)
thresholds_xz = [0.35] * (len(variances_xz.columns) - 1)

# Apply filtering for each plane and extract angles
filtered_variances_xy, common_angle_xy = filter_variances_per_plane(variances_xy, thresholds_xy)
filtered_variances_yz, common_angle_yz = filter_variances_per_plane(variances_yz, thresholds_yz)
filtered_variances_xz, common_angle_xz = filter_variances_per_plane(variances_xz, thresholds_xz)



# Display common angles and variances for each plane
print("Common angles for XY plane:", common_angle_xy)
print("Filtered variances for XY plane:\n", filtered_variances_xy)

print("Common angles for YZ plane:", common_angle_yz)
print("Filtered variances for YZ plane:\n", filtered_variances_yz)

print("Common angles for XZ plane:", common_angle_xz)
print("Filtered variances for XZ plane:\n", filtered_variances_xz)



# Function to calculate covariance for the given angles and rotation plane
def calculate_sum_covariance(data, angles, rotation_func):
    covariances = []
    for angle in angles:
        rotated_data = rotate_data_plane(data, rotation_func, angle)
        # Compute covariance matrix for each triplet and sum the values
        covariance_sum = 0
        for triplet in triplets:
            triplet_data = rotated_data[triplet]
            covariance_matrix = triplet_data.cov().values  # covariance matrix of the triplet
            covariance_sum += covariance_matrix.sum()  # sum of all elements in the covariance matrix
        covariances.append({"angle": angle, "sum_covariance": covariance_sum})
    return pd.DataFrame(covariances)



# Calculate sum of covariances for each plane using the common angles
sum_covariance_xy = calculate_sum_covariance(NOS2R_data, common_angle_xy, r_xy)
sum_covariance_yz = calculate_sum_covariance(NOS2R_data, common_angle_yz, r_yz)
sum_covariance_xz = calculate_sum_covariance(NOS2R_data, common_angle_xz, r_xz)



# Find the maximum sum of covariance and corresponding angle for each plane
covariance_sums = {
    'XY': [(angle, sum) for angle, sum in variances_xy.groupby('angle')['covariance_sum'].sum().items()],
    'YZ': [(angle, sum) for angle, sum in variances_yz.groupby('angle')['covariance_sum'].sum().items()],
    'XZ': [(angle, sum) for angle, sum in variances_xz.groupby('angle')['covariance_sum'].sum().items()]
}

# Find the plane and angle with the maximum covariance sum
max_covariance_info = max(
    [(plane, angle, cov_sum) for plane, data in covariance_sums.items() for angle, cov_sum in data],
    key=lambda x: x[2]
)

max_plane, max_angle, max_covariance_sum = max_covariance_info

print(f"Max Covariance Sum found in Plane: {max_plane}, Angle: {max_angle}, Value: {max_covariance_sum}")





# Rotate the dataset with the identified maximum angle and plane
def rotate_data_with_max_covariance(NOS2R_data, max_plane, max_angle):
    # Select rotation function based on max_plane
    rotation_func = {'XY': r_xy, 'YZ': r_yz, 'XZ': r_xz}[max_plane]
    
    # Convert angle to radians
    theta = np.radians(max_angle)

    # Define triplets from column names of NOS2R_data
    column_names = NOS2R_data.columns.tolist()
    triplets = [column_names[i:i+3] for i in range(0, len(column_names), 3)]

    # Create a copy of NOS2R_data for the rotated dataset
    rotated_data = NOS2R_data.copy()

    # Apply the rotation to each triplet in the dataset
    for triplet in triplets:
        for idx in range(len(rotated_data)):
            original_values = rotated_data.loc[idx, triplet].to_numpy().reshape(3, 1)
            rotated_values = np.dot(rotation_func(theta), original_values).flatten()
            rotated_data.loc[idx, triplet] = rotated_values

    # Add columns for angle and plane used in rotation
    rotated_data['angle'] = max_angle  # Angle in degrees
    rotated_data['rotation_plane'] = max_plane

    # Display the rotated data
    print(f"Rotated dataset for plane {max_plane} at angle {max_angle} degrees:")
    print(rotated_data.head())

    # Save the rotated dataset to CSV
    rotated_data.to_csv(f'rotated_{max_plane.lower()}_{max_angle}_data.csv', index=False)

    return rotated_data



rotated_data = rotate_data_with_max_covariance(NOS2R_data, max_plane, max_angle)
