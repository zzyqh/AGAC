# AGAC
Accurately Clustering Single-cell RNA-seq data by 
fusing node attribute features and graph structural features adaptively.
The experiment used 13 datasets, many of which were large and inconvenient to transfer. All datasets have been packaged into a zip package and uploaded. And you can extract it to the data folder to continue running.

# Requirement
Python 3.6.9
PyTorch 1.1.0

# Preprocessing 
 For the simulated datasets, we normalized
them using transcripts per million (TPM) method and then
scaled the value of each gene to [0, 1]. For real datasets, we
employed the procedure suggested by Seurat3.0 to normalize
and select top 2000 highly variable genes for scRNA-seq data,
then to scale the value of each gene to [0,1]. Note that for real
datasets normalized by FPKM, we first converted them to TPM.

# Usage
```
python AGAC.py --name [Chung|Biase]
```
