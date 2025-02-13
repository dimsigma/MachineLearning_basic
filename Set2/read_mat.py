import numpy as np
import scipy.io
import json

mat21 = scipy.io.loadmat("data21.mat")
mat22 = scipy.io.loadmat("data22.mat")
mat23 = scipy.io.loadmat("data23.mat")

matA1 = mat21["A_1"]   
matA2 = mat21["A_2"]
matB1 = mat21["B_1"]
matB2 = mat21["B_2"]

matXi_2 = mat22["X_i"]
matXn_2 = mat22["X_n"]

matXi_3 = mat23["X_i"]
matXn_3 = mat23["X_n"]

open("A1.json", 'w').close()
open("A2.json", 'w').close()
open("B1.json", 'w').close()
open("B2.json", 'w').close()

open("Xi2.json", 'w').close()
open("Xn2.json", 'w').close()

open("Xi3.json", 'w').close()
open("Xn3.json", 'w').close()

with open("A1.json", 'a') as f: json.dump(np.array(matA1).tolist(), f)
with open("A2.json", 'a') as f: json.dump(np.array(matA2).tolist(), f)
with open("B1.json", 'a') as f: json.dump(np.array(matB1).tolist(), f)
with open("B2.json", 'a') as f: json.dump(np.array(matB2).tolist(), f)

with open("Xi2.json", 'a') as f: json.dump(np.array(matXi_2).tolist(), f)
with open("Xn2.json", 'a') as f: json.dump(np.array(matXn_2).tolist(), f)

with open("Xi3.json", 'a') as f: json.dump(np.array(matXi_3).tolist(), f)
with open("Xn3.json", 'a') as f: json.dump(np.array(matXn_3).tolist(), f)