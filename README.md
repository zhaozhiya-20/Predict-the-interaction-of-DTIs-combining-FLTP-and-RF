# Predict-the-interaction-of-DTIs-combining-FLTP-and-RF
This project includes the codes of new model and comparison model,and also provides PSSM matrix of fltp descriptor, drug molecular fingerprint and protein sequence.
1. We provide the software of FLTP and RF to assist constructiing the proposed model.
2. We provide the codes of ZMs, LGBM, and SVM to help constructing the comparison.
3. We provide the 881-dimentional fingerprints of drugs and the PSSM matrixs of protein sequences. As can be noted that the full data contains the label, FLTPs discribers of PSSMs, and fingerprints. The first column in the CSV file is the label，Columns 1 to 512 in the CSV file are FLTP features，the rest columns represents the 881-dimentional fingerprints of drugs.
