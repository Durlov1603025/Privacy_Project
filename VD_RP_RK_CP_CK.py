import numpy as np
import pandas as pd

def VD_RP_RK_CP_CK(orgData, perturbed_data, numOFfeatures, totPtrn):
    # VD calculation (Vector Difference using Frobenius norm)
    org = orgData
    chngd = perturbed_data
    VD = np.linalg.norm(np.abs(org - chngd), 'fro') / np.linalg.norm(np.abs(org), 'fro')

    sumRP = 0
    countRK = 0
    countCK = 0

    # Rank Profile and Rank K calculations
    for i in range(numOFfeatures):
        # Sort and rank original data
        indxOrg = np.argsort(org[:, i], axis=0)
        rankOrg = np.arange(1, len(org[:, i]) + 1)
        rankOrg[indxOrg] = rankOrg

        # Sort and rank changed data
        indxChngd = np.argsort(chngd[:, i], axis=0)
        rankChngd = np.arange(1, len(chngd[:, i]) + 1)
        rankChngd[indxChngd] = rankChngd

        # Sum of rank profile differences (RP)
        sumRP += np.sum(np.abs(rankOrg - rankChngd))

        # Counting exact rank matches (RK)
        for j in range(totPtrn):
            if rankOrg[j] == rankChngd[j]:
                countRK += 1

    # Average values of original and changed data
    avgOrg = np.sum(org, axis=0) / numOFfeatures
    avgChngd = np.sum(chngd, axis=0) / numOFfeatures

    # Rank average values
    dAvgOrg = avgOrg
    dAvgChngd = avgChngd

    # Rank sorting and calculating CP (Combined Profile)
    indxAvgOrg = np.argsort(dAvgOrg, axis=0)
    rankAvgOrg = np.arange(1, len(dAvgOrg) + 1)
    rankAvgOrg[indxAvgOrg] = rankAvgOrg

    indxAvgChngd = np.argsort(dAvgChngd, axis=0)
    rankAvgChngd = np.arange(1, len(dAvgChngd) + 1)
    rankAvgChngd[indxAvgChngd] = rankAvgChngd

    # Calculate CP
    diffCP = np.sum(np.abs(rankAvgOrg - rankAvgChngd))

    # Counting exact average rank matches (CK)
    for j in range(numOFfeatures):
        if rankAvgOrg[j] == rankAvgChngd[j]:
            countCK += 1

    # Normalizing results
    RP = sumRP / (numOFfeatures * totPtrn)
    RK = countRK / (numOFfeatures * totPtrn)
    CP = diffCP / numOFfeatures
    CK = countCK / numOFfeatures

    # Return the result as a numpy array (VD, RP, RK, CP, CK)
    resultMetric = np.array([VD, RP, RK, CP, CK])

    return resultMetric





#Calculation for NOS2R
orgData = pd.read_csv('original_data.csv')
perturbed_data1 = pd.read_csv('NOS2R_data.csv')
numOFfeatures = orgData.shape[1] 
totPtrn = orgData.shape[0] 

result_NOS2R = VD_RP_RK_CP_CK(orgData, perturbed_data1, numOFfeatures, totPtrn)



#Calculation for NOS2R2
orgData = pd.read_csv('original_data.csv')
perturbed_data2 = pd.read_csv('NOS2R2_data.csv')
result_NOS2R2 = VD_RP_RK_CP_CK(orgData, perturbed_data2, numOFfeatures, totPtrn)