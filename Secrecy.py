import numpy as np
import pandas as pd


# Define initial values
orgData = pd.read_csv('original_data.csv')
NOS2R_data = pd.read_csv('NOS2R_data.csv') 
NOS2R2_data = pd.read_csv('NOS2R2_data.csv') 
numOFfeatures = orgData.shape[1]

# NOS2R calculation
SecurityRot1 = 0
for k in range(numOFfeatures):
    SecurityRot1 += np.var(orgData[:, k] - NOS2R_data[:, k]) / np.var(orgData[:, k])
privacyRotation1 = SecurityRot1 / numOFfeatures
Secrecy_NOS2R = [privacyRotation1]

# NOS2R2 calculation
SecurityRot2 = 0
for j in range(numOFfeatures):
    SecurityRot2 += np.var(orgData[:, j] - NOS2R2_data[:, j]) / np.var(orgData[:, j])
privacyRotation2 = SecurityRot2 / numOFfeatures
Secrecy_NOS2R2 = [privacyRotation2]